"""Create the deterministic time-shifted ImBrAid processing copy."""

from __future__ import annotations

import calendar
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR
import json
import os
from pathlib import Path
import re
import shutil
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET


TARGET_ORIGIN_EPOCH_S = 946684800
SILAB_TIME_COLUMNS = {"date", "systemtimehhmm", "systemunixtimeseconds"}
SILAB_XDF_VALUE_BYTES = 8
XDF_DATETIME_ELEMENT_RE = re.compile(
    rb"(<datetime(?:\s[^>]{0,})?>)([^<]+)(</datetime\s*>)",
    re.IGNORECASE,
)
XDF_DATETIME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})T"
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?P<fraction>\.\d{1,6})?"
    r"(?P<zone>Z|z|[+-]\d{2}:?\d{2})$"
)


@dataclass(frozen=True)
class TimeShiftEntry:
    source_path: Path
    relative_path: Path
    action: str
    delta_seconds: int | None


@dataclass(frozen=True)
class TimeShiftPlan:
    records: int
    entries: tuple[TimeShiftEntry, ...]


def _files(root: Path) -> Iterable[Path]:
    for directory, names, files in os.walk(root):
        names.sort(key=str.casefold)
        files.sort(key=str.casefold)
        for name in files:
            yield Path(directory) / name


def _raw_csv_fields(line: bytes) -> list[tuple[int, int]]:
    fields = []
    start = 0
    quoted = False
    index = 0
    while index < len(line):
        value = line[index]
        if value == 34:
            if quoted and index + 1 < len(line) and line[index + 1] == 34:
                index += 2
                continue
            quoted = not quoted
        elif value == 44 and not quoted:
            fields.append((start, index))
            start = index + 1
        index += 1
    fields.append((start, len(line)))
    return fields


def _line_parts(line: bytes) -> tuple[bytes, bytes]:
    body = line.rstrip(b"\r\n")
    return body, line[len(body):]


def _timestamp_index(header: bytes) -> int:
    names = next(csv.reader([header.decode("utf-8-sig")]))
    return [name.strip().casefold() for name in names].index("timestamp")


def _decimal_field(field: bytes) -> tuple[Decimal, bool]:
    quoted = len(field) >= 2 and field[:1] == b'"' and field[-1:] == b'"'
    token = field[1:-1] if quoted else field
    return Decimal(token.decode("ascii")), quoted


def _idun_first_timestamp(path: Path) -> tuple[int, Decimal | None]:
    with path.open("rb") as handle:
        header = _line_parts(handle.readline())[0]
        index = _timestamp_index(header)
        for line in handle:
            body = _line_parts(line)[0]
            if body:
                start, stop = _raw_csv_fields(body)[index]
                return index, _decimal_field(body[start:stop])[0]
    return index, None


def _decimal_text(value: Decimal, delta_seconds: int) -> bytes:
    places = max(0, -value.as_tuple().exponent)
    shifted = value + Decimal(delta_seconds)
    return format(shifted, f".{places}f").encode("ascii")


def _shift_idun_csv(
    source_path: Path,
    output_path: Path,
    timestamp_index: int,
    delta_seconds: int,
) -> None:
    with source_path.open("rb") as source, output_path.open("wb") as output:
        output.write(source.readline())
        for line in source:
            body, ending = _line_parts(line)
            if not body:
                output.write(line)
                continue
            fields = _raw_csv_fields(body)
            start, stop = fields[timestamp_index]
            value, quoted = _decimal_field(body[start:stop])
            replacement = _decimal_text(value, delta_seconds)
            if quoted:
                replacement = b'"' + replacement + b'"'
            output.write(body[:start] + replacement + body[stop:] + ending)


def _parse_xdf_datetime(value: str) -> datetime:
    match = XDF_DATETIME_RE.fullmatch(value)
    zone = match.group("zone")
    normalized_zone = "+00:00" if zone.casefold() == "z" else zone
    if len(normalized_zone) == 5:
        normalized_zone = normalized_zone[:3] + ":" + normalized_zone[3:]
    fraction = match.group("fraction") or ""
    return datetime.fromisoformat(
        f"{match.group('date')}T{match.group('time')}{fraction}{normalized_zone}"
    )


def _datetime_epoch(value: str) -> Decimal:
    parsed = _parse_xdf_datetime(value).astimezone(timezone.utc)
    seconds = calendar.timegm(parsed.utctimetuple())
    return Decimal(seconds) + Decimal(parsed.microsecond) / Decimal(1_000_000)


def _shift_datetime(value: str, delta_seconds: int) -> str:
    match = XDF_DATETIME_RE.fullmatch(value)
    shifted = _parse_xdf_datetime(value) + timedelta(seconds=delta_seconds)
    fraction = match.group("fraction")
    if fraction:
        digits = len(fraction) - 1
        fraction = "." + f"{shifted.microsecond:06d}"[:digits]
    else:
        fraction = ""
    return shifted.strftime("%Y-%m-%dT%H:%M:%S") + fraction + match.group("zone")


def _read_xdf_header(path: Path) -> str:
    with path.open("rb") as handle:
        handle.read(4)
        width = handle.read(1)[0]
        length_bytes = handle.read(width)
        length = int.from_bytes(length_bytes, "little")
        handle.read(2)
        payload = handle.read(length - 2)
    match = XDF_DATETIME_ELEMENT_RE.search(payload)
    return match.group(2).decode("utf-8")


def _xdf_varlen(value: int) -> bytes:
    width = 1 if value < 256 else 4 if value < 2**32 else 8
    return bytes((width,)) + value.to_bytes(width, "little")


def _rewrite_silab_stream_header(payload: bytes):
    root = ET.fromstring(payload[4:])
    if root.findtext("name").strip().casefold() != "silab":
        return payload, None

    channels = root.find("desc/channels")
    channel_nodes = channels.findall("channel")
    drop_indices = tuple(
        index
        for index, channel in enumerate(channel_nodes)
        if channel.findtext("label").strip().casefold() in SILAB_TIME_COLUMNS
    )
    if not drop_indices:
        return payload, None

    channel_count = int(root.findtext("channel_count"))
    for index in reversed(drop_indices):
        channels.remove(channel_nodes[index])
    root.find("channel_count").text = str(channel_count - len(drop_indices))
    stream_id = int.from_bytes(payload[:4], "little")
    rewritten = payload[:4] + ET.tostring(root, encoding="utf-8")
    return rewritten, (stream_id, channel_count, drop_indices)


def _rewrite_silab_samples(
    payload: bytes,
    channel_count: int,
    drop_indices: Sequence[int],
) -> bytes:
    count_width = payload[4]
    position = 5 + count_width
    sample_count = int.from_bytes(payload[5:position], "little")
    rewritten = bytearray(payload[:position])
    ranges = []
    start = 0
    for index in drop_indices:
        if start < index:
            ranges.append(
                (start * SILAB_XDF_VALUE_BYTES, index * SILAB_XDF_VALUE_BYTES)
            )
        start = index + 1
    if start < channel_count:
        ranges.append(
            (start * SILAB_XDF_VALUE_BYTES, channel_count * SILAB_XDF_VALUE_BYTES)
        )

    source = memoryview(payload)
    sample_bytes = channel_count * SILAB_XDF_VALUE_BYTES
    for _ in range(sample_count):
        timestamp_end = position + 1 + payload[position]
        rewritten.extend(source[position:timestamp_end])
        value_end = timestamp_end + sample_bytes
        values = source[timestamp_end:value_end]
        for first, last in ranges:
            rewritten.extend(values[first:last])
        position = value_end
    return bytes(rewritten)


def _shift_xdf(source_path: Path, output_path: Path, delta_seconds: int) -> None:
    silab_streams = {}
    with source_path.open("rb") as source, output_path.open("wb") as output:
        output.write(source.read(4))
        while width_bytes := source.read(1):
            length_bytes = source.read(width_bytes[0])
            length = int.from_bytes(length_bytes, "little")
            tag = source.read(2)
            payload = source.read(length - 2)
            rewritten = False

            if tag == b"\x01\x00":
                match = XDF_DATETIME_ELEMENT_RE.search(payload)
                value = match.group(2).decode("utf-8")
                replacement = _shift_datetime(value, delta_seconds).encode("utf-8")
                payload = payload[:match.start(2)] + replacement + payload[match.end(2):]
                rewritten = True
            elif tag == b"\x02\x00":
                payload, stream = _rewrite_silab_stream_header(payload)
                if stream is not None:
                    stream_id, channel_count, drop_indices = stream
                    silab_streams[stream_id] = channel_count, drop_indices
                    rewritten = True
            elif tag == b"\x03\x00":
                stream_id = int.from_bytes(payload[:4], "little")
                if stream_id in silab_streams:
                    payload = _rewrite_silab_samples(payload, *silab_streams[stream_id])
                    rewritten = True

            if rewritten:
                output.write(_xdf_varlen(len(payload) + 2) + tag + payload)
            else:
                output.write(width_bytes + length_bytes + tag + payload)


def _shift_et_recording_created(
    source_path: Path,
    output_path: Path,
    delta_seconds: int,
) -> None:
    value = json.loads(source_path.read_text(encoding="utf-8"))
    value["created"] = _shift_datetime(value["created"], delta_seconds)
    output_path.write_text(
        json.dumps(value, separators=(",", ":")),
        encoding="utf-8",
    )


def _silab_drop_indices(path: Path) -> tuple[int, ...]:
    with path.open("rb") as handle:
        header = _line_parts(handle.readline())[0]
    names = next(csv.reader([header.decode("utf-8-sig")]))
    return tuple(
        index
        for index, name in enumerate(names)
        if name.strip().casefold() in SILAB_TIME_COLUMNS
    )


def _drop_silab_columns(
    source_path: Path,
    output_path: Path,
    drop_indices: Sequence[int],
) -> None:
    dropped = set(drop_indices)
    with source_path.open("rb") as source, output_path.open("wb") as output:
        for line in source:
            body, ending = _line_parts(line)
            fields = _raw_csv_fields(body)
            retained = [
                body[start:stop]
                for index, (start, stop) in enumerate(fields)
                if index not in dropped
            ]
            output.write(b",".join(retained) + ending)


def build_time_shift_plan(source_root: str | Path) -> TimeShiftPlan:
    source_root = Path(source_root)
    entries = []
    record_count = 0
    for record_root in sorted(
        (path for path in source_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    ):
        record_count += 1
        files = tuple(_files(record_root))
        idun = {}
        xdf = {}
        clocks = []
        for path in files:
            relative = path.relative_to(source_root)
            modality = relative.parts[1].casefold()
            if modality == "idun" and path.suffix.casefold() == ".csv":
                index, first = _idun_first_timestamp(path)
                idun[path] = (index, first)
                if first is not None:
                    clocks.append(first)
            elif modality == "lsl" and path.suffix.casefold() == ".xdf":
                value = _read_xdf_header(path)
                xdf[path] = value
                clocks.append(_datetime_epoch(value))
        delta = None
        if clocks:
            anchor = min(clocks).to_integral_value(rounding=ROUND_FLOOR)
            delta = TARGET_ORIGIN_EPOCH_S - int(anchor)

        for path in files:
            relative = path.relative_to(source_root)
            modality = relative.parts[1].casefold()
            if path in idun and idun[path][1] is not None:
                action = "shift_idun_unix_timestamp"
            elif path in idun:
                action = "copy_empty_idun_csv"
            elif path in xdf:
                action = "shift_xdf_and_drop_silab_time_channels"
            elif (
                modality == "et"
                and path.name.casefold() == "recording.g3"
                and delta is not None
            ):
                action = "shift_et_recording_created"
            elif (
                modality == "silab"
                and path.suffix.casefold() == ".asc"
                and _silab_drop_indices(path)
            ):
                action = "drop_silab_wall_clock_columns"
            else:
                action = "copy_unchanged"
            entries.append(TimeShiftEntry(path, relative, action, delta))
    return TimeShiftPlan(record_count, tuple(entries))


def summarize_time_shift(plan: TimeShiftPlan) -> dict[str, object]:
    return {
        "records": plan.records,
        "files": len(plan.entries),
        "actions": dict(sorted(Counter(entry.action for entry in plan.entries).items())),
    }


def execute_time_shift(plan: TimeShiftPlan, output_root: str | Path) -> dict[str, object]:
    output_root = Path(output_root)
    for entry in plan.entries:
        target = output_root / entry.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry.action == "shift_idun_unix_timestamp":
            index = _idun_first_timestamp(entry.source_path)[0]
            _shift_idun_csv(entry.source_path, target, index, entry.delta_seconds)
        elif entry.action == "shift_xdf_and_drop_silab_time_channels":
            _shift_xdf(entry.source_path, target, entry.delta_seconds)
        elif entry.action == "shift_et_recording_created":
            _shift_et_recording_created(entry.source_path, target, entry.delta_seconds)
        elif entry.action == "drop_silab_wall_clock_columns":
            _drop_silab_columns(entry.source_path, target, _silab_drop_indices(entry.source_path))
        else:
            shutil.copy2(entry.source_path, target)
    return summarize_time_shift(plan)
