"""Copy the ImBrAid source tree with anonymized participant and visit names."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
import re
import secrets
import shutil
from typing import Callable, Iterable, Mapping, Sequence

from helpers.project_config import load_project_config


IDENTITY_RE = re.compile(r"(?P<pid>T\d{3})_(?P<visit>V\d+)", re.IGNORECASE)
DEFAULT_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
TEXT_SUFFIXES = {".asc", ".cfg", ".csv", ".num", ".tsv", ".txt"}


@dataclass(frozen=True, order=True)
class RecordKey:
    pid: str
    visit: str

    @property
    def pid_visit(self) -> str:
        return f"{self.pid}_{self.visit}"

    @classmethod
    def parse(cls, value: str) -> "RecordKey":
        match = IDENTITY_RE.fullmatch(value.strip())
        if match is None:
            raise ValueError(f"Cannot read participant and visit from {value!r}")
        return cls(match.group("pid").upper(), match.group("visit").upper())


@dataclass(frozen=True)
class AnonymizationConfig:
    source_root: Path
    output_root: Path
    time_shift_output_root: Path
    mapping_path: Path
    excluded_records: frozenset[RecordKey]
    random_id_length: int
    random_id_alphabet: str


@dataclass(frozen=True)
class SourceEntry:
    source_path: Path
    relative_path: Path
    record: RecordKey

    @property
    def size(self) -> int:
        return self.source_path.stat().st_size


@dataclass(frozen=True)
class EtRecording:
    relative_path: Path
    record: RecordKey


@dataclass(frozen=True)
class ExportPlanEntry:
    source: SourceEntry
    random_id: str
    target_relative: Path


@dataclass(frozen=True)
class GeneratedParticipantEntry:
    random_id: str
    target_relative: Path




def load_config(path: str | Path) -> AnonymizationConfig:
    config = load_project_config(path)
    values = config.values.get("anonymization", {})
    random_id = values.get("random_id", {})
    return AnonymizationConfig(
        source_root=config.path("paths", "raw_data_root"),
        output_root=config.path("paths", "random_id_root"),
        time_shift_output_root=config.path("paths", "processing_data_root"),
        mapping_path=config.path("files", "private_mapping_csv"),
        excluded_records=frozenset(
            RecordKey.parse(value) for value in values.get("excluded_pid_visits", [])
        ),
        random_id_length=int(random_id.get("length", 6)),
        random_id_alphabet=str(random_id.get("alphabet", DEFAULT_ALPHABET)),
    )


def _files(root: Path) -> Iterable[Path]:
    for directory, names, files in os.walk(root):
        names.sort(key=str.casefold)
        files.sort(key=str.casefold)
        for name in files:
            yield Path(directory) / name


def _record_from_path(path: Path) -> RecordKey | None:
    match = IDENTITY_RE.search(path.as_posix())
    if match is None:
        return None
    return RecordKey(match.group("pid").upper(), match.group("visit").upper())


def discover_source_entries(config: AnonymizationConfig) -> tuple[SourceEntry, ...]:
    entries = []
    for path in _files(config.source_root):
        relative = path.relative_to(config.source_root)
        record = _record_from_path(relative)
        if record is not None and record not in config.excluded_records:
            entries.append(SourceEntry(path, relative, record))
    return tuple(entries)


def discover_et_recordings(config: AnonymizationConfig) -> tuple[EtRecording, ...]:
    recordings = []
    for directory, names, files in os.walk(config.source_root):
        names.sort(key=str.casefold)
        relative = Path(directory).relative_to(config.source_root)
        if len(relative.parts) == 3 and relative.parts[1].casefold() == "et":
            record = _record_from_path(relative)
            if record is not None and record not in config.excluded_records:
                recordings.append(EtRecording(relative, record))
    return tuple(recordings)


def load_mapping(path: str | Path) -> dict[RecordKey, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return {
            RecordKey.parse(row["pid_visit"]): row["random_id"]
            for row in csv.DictReader(handle)
        }


def _new_random_id(
    length: int,
    alphabet: str,
    used: set[str],
    chooser: Callable[[Sequence[str]], str],
) -> str:
    while True:
        value = "".join(chooser(alphabet) for _ in range(length))
        if value not in used:
            return value


def create_or_load_mapping(
    config: AnonymizationConfig,
    records: Iterable[RecordKey],
    *,
    commit: bool = False,
    chooser: Callable[[Sequence[str]], str] = secrets.choice,
) -> dict[RecordKey, str]:
    mapping = load_mapping(config.mapping_path) if config.mapping_path.exists() else {}
    used = set(mapping.values())
    for record in sorted(set(records)):
        if record not in mapping:
            mapping[record] = _new_random_id(
                config.random_id_length,
                config.random_id_alphabet,
                used,
                chooser,
            )
            used.add(mapping[record])
    if commit:
        config.mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with config.mapping_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["pid_visit", "random_id"])
            for record in sorted(mapping):
                writer.writerow([record.pid_visit, mapping[record]])
    return mapping


def _replace(value: str, record: RecordKey, random_id: str) -> str:
    value = re.sub(re.escape(record.pid_visit), f"{random_id}_VX", value, flags=re.IGNORECASE)
    value = re.sub(re.escape(record.pid), random_id, value, flags=re.IGNORECASE)
    return re.sub(
        rf"(?<![A-Za-z0-9]){re.escape(record.visit)}(?![A-Za-z0-9])",
        "VX",
        value,
        flags=re.IGNORECASE,
    )


def _target_relative(entry: SourceEntry, random_id: str) -> Path:
    parts = [_replace(part, entry.record, random_id) for part in entry.relative_path.parts]
    parts[0] = random_id
    return Path(*parts)


def _canonical_label(modality: str, name: str, is_directory: bool) -> tuple[str, bool]:
    lowered = modality.casefold()
    if lowered == "driver video":
        return "driver_video", False
    if lowered == "et":
        if is_directory:
            return "et_raw", False
        if Path(name).suffix.casefold() in (".avi", ".mkv", ".mp4", ".webm"):
            return "eyeball_video", False
        return "et", False
    if lowered == "lsl":
        return "lsl", False
    if lowered == "silab":
        return "silab", False
    if lowered == "idun":
        stem = Path(name).stem.casefold()
        for label in ("eeg", "imu", "quality"):
            if re.search(rf"(?<![a-z0-9]){label}(?![a-z0-9])", stem):
                return label, label == "eeg"
        if Path(name).suffix.casefold() == ".pdf":
            return "report", False
        return "idun", False
    return lowered.replace(" ", "_"), False


def _canonical_assignments(
    entries: Sequence[SourceEntry],
    recordings: Sequence[EtRecording],
    mapping: Mapping[RecordKey, str],
) -> dict[Path, Path]:
    roots: dict[Path, tuple[RecordKey, str, bool]] = {}
    for entry in entries:
        parts = entry.relative_path.parts
        if len(parts) >= 3:
            root = Path(parts[0]) / parts[1] / parts[2]
            roots[root] = (entry.record, parts[1], len(parts) > 3)
    for recording in recordings:
        roots[recording.relative_path] = (recording.record, "ET", True)

    groups: dict[tuple[RecordKey, str, str, bool, bool], list[Path]] = {}
    for root, (record, modality, is_directory) in roots.items():
        label, always_numbered = _canonical_label(
            modality,
            root.name,
            is_directory,
        )
        key = (record, modality, label, is_directory, always_numbered)
        groups.setdefault(key, []).append(root)

    assignments = {}
    for key, source_roots in groups.items():
        record, modality, label, is_directory, always_numbered = key
        random_id = mapping[record]
        for index, source_root in enumerate(
            sorted(source_roots, key=lambda path: path.as_posix().casefold())
        ):
            number = index + 1 if always_numbered else index
            suffix = f"_{number}" if number else ""
            extension = "" if is_directory else source_root.suffix
            target_name = f"{random_id}_VX_{label}{suffix}{extension}"
            assignments[source_root] = Path(random_id) / modality / target_name
    return assignments


def _canonical_target(
    entry: SourceEntry,
    random_id: str,
    assignments: Mapping[Path, Path],
) -> Path:
    parts = entry.relative_path.parts
    if len(parts) < 3:
        return _target_relative(entry, random_id)
    source_root = Path(parts[0]) / parts[1] / parts[2]
    target = assignments[source_root]
    for part in parts[3:]:
        target /= _replace(part, entry.record, random_id)
    return target


def build_export_plan(
    entries: Iterable[SourceEntry],
    mapping: Mapping[RecordKey, str],
    recordings: Sequence[EtRecording] = (),
) -> tuple[ExportPlanEntry, ...]:
    entries = tuple(entries)
    assignments = _canonical_assignments(entries, recordings, mapping)
    plan = []
    for entry in entries:
        random_id = mapping[entry.record]
        target = _canonical_target(entry, random_id, assignments)
        plan.append(ExportPlanEntry(entry, random_id, target))
    return tuple(plan)


def build_generated_participant_plan(
    recordings: Sequence[EtRecording],
    plan: Sequence[ExportPlanEntry],
    mapping: Mapping[RecordKey, str],
) -> tuple[GeneratedParticipantEntry, ...]:
    existing = {entry.target_relative.as_posix().casefold() for entry in plan}
    assignments = _canonical_assignments(
        tuple(entry.source for entry in plan),
        recordings,
        mapping,
    )
    generated = []
    for recording in recordings:
        target = assignments[recording.relative_path] / "meta" / "participant"
        key = target.as_posix().casefold()
        if key not in existing:
            generated.append(GeneratedParticipantEntry(
                mapping[recording.record],
                target,
            ))
    return tuple(generated)


def summarize_plan(
    entries: Sequence[SourceEntry],
    plan: Sequence[ExportPlanEntry] | None = None,
    generated_participants: Sequence[GeneratedParticipantEntry] = (),
    additional_records: Iterable[RecordKey] = (),
) -> dict[str, object]:
    modalities: dict[str, dict[str, int]] = {}
    for entry in entries:
        modality = entry.relative_path.parts[1]
        bucket = modalities.setdefault(modality, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += entry.size
    summary: dict[str, object] = {
        "records": len({entry.record for entry in entries} | set(additional_records)),
        "files": len(entries),
        "bytes": sum(entry.size for entry in entries),
        "modalities": dict(sorted(modalities.items())),
    }
    if plan is not None:
        summary["generated_participant_files"] = len(generated_participants)
        summary["target_paths"] = len(plan) + len(generated_participants)
    return summary


def _copy(entry: ExportPlanEntry, output_root: Path) -> None:
    source = entry.source.source_path
    target = output_root / entry.target_relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.name.casefold() == "participant":
        target.write_text(
            json.dumps({"name": entry.random_id}, separators=(",", ":")),
            encoding="utf-8",
        )
    elif source.suffix.casefold() in (".g3", ".json"):
        value = json.loads(source.read_text(encoding="utf-8"))

        def replace_json(item):
            if isinstance(item, dict):
                return {
                    _replace(str(key), entry.source.record, entry.random_id): replace_json(nested)
                    for key, nested in item.items()
                }
            if isinstance(item, list):
                return [replace_json(nested) for nested in item]
            if isinstance(item, str):
                return _replace(item, entry.source.record, entry.random_id)
            return item

        value = replace_json(value)
        if (
            entry.source.relative_path.parts[1].casefold() == "et"
            and source.name.casefold() == "recording.g3"
        ):
            value["name"] = f"{entry.random_id}_VX"
        target.write_text(
            json.dumps(value, separators=(",", ":")),
            encoding="utf-8",
        )
    elif source.suffix.casefold() in TEXT_SUFFIXES:
        with source.open(
            "r", encoding="utf-8", errors="surrogateescape", newline=""
        ) as input_file:
            text = input_file.read()
        with target.open(
            "w", encoding="utf-8", errors="surrogateescape", newline=""
        ) as output_file:
            output_file.write(_replace(text, entry.source.record, entry.random_id))
    else:
        shutil.copy2(source, target)


def execute_export(
    config: AnonymizationConfig,
    plan: Sequence[ExportPlanEntry],
    generated_participants: Sequence[GeneratedParticipantEntry] = (),
) -> dict[str, object]:
    for entry in plan:
        _copy(entry, config.output_root)
    for entry in generated_participants:
        target = config.output_root / entry.target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"name": entry.random_id}, separators=(",", ":")),
            encoding="utf-8",
        )
    return {
        "records": len(
            {entry.random_id for entry in plan}
            | {entry.random_id for entry in generated_participants}
        ),
        "files": len(plan) + len(generated_participants),
        "generated_participant_files": len(generated_participants),
    }
