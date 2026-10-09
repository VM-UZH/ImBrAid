"""Exact Driver PTS persistence and compact ImBrAid publication export.

The private 00.5 path opens raw Driver videos only to obtain exact relative
presentation timestamps. Step 10 consumes that exact-PTS landmark CSV and the
current affine mappings; it never discovers or opens raw videos.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..DriverVideo import driver_landmarks
from . import driver_lsl_alignment as driver_lsl
from . import panel_annotation as panel
from .publication_contract import required_publication_columns


SUMMARY_COLUMNS = [
    "project",
    "participant",
    "visit",
    "segment_id",
    "video_path",
    "landmark_csv_path",
    "output_path",
    "status",
    "frames",
    "driver_et_mapping_status",
    "et_lsl_mapping_status",
    "et_raw_time_count",
    "lsl_time_count",
    "error",
]

EXACT_PTS_SUMMARY_COLUMNS = [
    "participant",
    "video_path",
    "landmark_csv_path",
    "status",
    "frames",
    "error",
]

PUBLICATION_DATA_COLUMNS = list(required_publication_columns("DriverVideo"))


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _read_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        header = next(csv.reader(csv_file), None)
    if header is None:
        raise ValueError(f"CSV is empty: {csv_path}")
    return header


def _read_mapping_table(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path).expanduser().resolve()
    table = pd.read_csv(path, dtype=str)
    if table.empty:
        raise ValueError(f"Mapping table contains no rows: {path}")
    return table


def _format_time(value: float) -> str:
    return "" if not math.isfinite(value) else format(value, ".15g")


def _landmark_stem(landmark_path: Path) -> str:
    suffix = driver_landmarks.DEFAULT_OUTPUT_SUFFIX
    if not landmark_path.name.endswith(suffix):
        raise ValueError(
            f"Landmark CSV name does not end with {suffix!r}: "
            f"{landmark_path.name}"
        )
    stem = landmark_path.name[: -len(suffix)]
    if not stem:
        raise ValueError(f"Landmark CSV has an empty video stem: {landmark_path}")
    return stem


def _landmark_identity(landmark_path: Path) -> tuple[str, str, str]:
    """Use only the first two filename tokens as participant and visit."""

    stem = _landmark_stem(landmark_path)
    tokens = stem.split("_")
    if len(tokens) < 2 or not tokens[0] or not tokens[1]:
        raise ValueError(
            "Driver landmark filename must contain participant and visit as its "
            f"first two underscore-separated tokens: {landmark_path.name}"
        )
    return stem, tokens[0], tokens[1]


def _publication_output_path(output_root: Path, landmark_path: Path) -> Path:
    stem = _landmark_stem(landmark_path)
    if stem.endswith("_driver_video"):
        return output_root / f"{stem}_synchronized.csv"
    return output_root / f"{stem}_driver_video_synchronized.csv"


def _identity_mask(
    table: pd.DataFrame,
    *,
    participant: str,
    visit: str,
    project: str | None,
) -> pd.Series:
    required = {"project", "pid", "visit"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Mapping table is missing identity columns: {missing}")

    mask = (
        table["pid"].astype("string").fillna("").str.strip().str.casefold()
        == participant.strip().casefold()
    ) & (
        table["visit"].astype("string").fillna("").str.strip().str.casefold()
        == visit.strip().casefold()
    )
    if project is not None and str(project).strip():
        mask &= (
            table["project"]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.casefold()
            == str(project).strip().casefold()
        )
    return mask


def _select_driver_mapping(
    alignments: pd.DataFrame,
    *,
    participant: str,
    visit: str,
    project: str | None,
) -> pd.Series:
    if "segment_id" not in alignments.columns:
        raise ValueError("Driver/ET mapping table is missing segment_id")
    matches = alignments.loc[
        _identity_mask(
            alignments,
            participant=participant,
            visit=visit,
            project=project,
        )
    ]
    if matches.empty:
        raise ValueError(
            "No Driver/ET mapping matches landmark identity "
            f"{participant!r}/{visit!r}"
        )
    if len(matches) != 1:
        segments = sorted(
            matches["segment_id"].astype("string").fillna("").str.strip()
        )
        raise ValueError(
            "Driver landmark identity matches multiple Driver/ET segments: "
            f"participant={participant!r}, visit={visit!r}, segments={segments}"
        )
    return matches.iloc[0]


def _select_et_lsl_mapping(
    alignments: pd.DataFrame,
    driver_mapping: pd.Series,
) -> pd.Series:
    project = _text(driver_mapping.get("project"))
    participant = _text(driver_mapping.get("pid"))
    visit = _text(driver_mapping.get("visit"))
    matches = alignments.loc[
        _identity_mask(
            alignments,
            participant=participant,
            visit=visit,
            project=project,
        )
    ]
    if matches.empty:
        raise ValueError(
            "No ET/LSL mapping matches Driver mapping identity "
            f"{project!r}/{participant!r}/{visit!r}"
        )
    if len(matches) != 1:
        raise ValueError(
            "More than one ET/LSL mapping matches Driver mapping identity "
            f"{project!r}/{participant!r}/{visit!r}"
        )
    return matches.iloc[0]


def _probe_exact_relative_pts(
    video_path: Path,
    *,
    ffprobe_path: str | Path | None,
    ffprobe_timeout_s: float,
) -> np.ndarray:
    frame_index = panel.probe_video_frame_index(
        video_path,
        ffprobe_path=ffprobe_path,
        timeout_s=ffprobe_timeout_s,
    )
    return np.asarray(frame_index.relative_times_s, dtype=float)


def _persist_exact_pts_one(
    video_path: Path,
    landmark_path: Path,
    *,
    overwrite: bool,
    ffprobe_path: str | Path | None,
    ffprobe_timeout_s: float,
) -> tuple[str, float | int]:
    if not landmark_path.is_file():
        raise FileNotFoundError(f"Landmark CSV does not exist: {landmark_path}")

    source_header = _read_header(landmark_path)
    exact_header = driver_landmarks.EXACT_PTS_CSV_COLUMNS
    if source_header == exact_header and not overwrite:
        return "skipped_existing", np.nan
    if source_header not in (driver_landmarks.CSV_COLUMNS, exact_header):
        raise ValueError(f"Landmark CSV has an unsupported schema: {landmark_path}")

    exact_times = _probe_exact_relative_pts(
        video_path,
        ffprobe_path=ffprobe_path,
        ffprobe_timeout_s=ffprobe_timeout_s,
    )
    base_indices = [
        source_header.index(column) for column in driver_landmarks.CSV_COLUMNS
    ]
    temporary_path = landmark_path.with_suffix(landmark_path.suffix + ".pts.part")
    temporary_path.unlink(missing_ok=True)

    try:
        with (
            landmark_path.open("r", newline="", encoding="utf-8") as source_file,
            temporary_path.open("w", newline="", encoding="utf-8") as target_file,
        ):
            reader = csv.reader(source_file)
            writer = csv.writer(target_file, lineterminator="\n")
            current_header = next(reader, None)
            if current_header != source_header:
                raise RuntimeError(f"Landmark CSV header changed: {landmark_path}")
            writer.writerow(exact_header)

            row_count = 0
            for source_row in reader:
                if len(source_row) != len(source_header):
                    raise ValueError(
                        f"Landmark row {row_count} has {len(source_row)} columns; "
                        f"expected {len(source_header)}"
                    )
                if row_count >= len(exact_times):
                    raise ValueError(
                        "Landmark CSV contains more rows than the ffprobe PTS index"
                    )
                base_row = [source_row[index] for index in base_indices]
                try:
                    source_frame_index = int(base_row[1])
                except ValueError as error:
                    raise ValueError(
                        f"Landmark row {row_count} has an invalid frame_index"
                    ) from error
                if source_frame_index != row_count:
                    raise ValueError(
                        f"Landmark row {row_count} has frame_index={source_frame_index}; "
                        f"expected {row_count}"
                    )
                writer.writerow(
                    [
                        *base_row[:2],
                        _format_time(float(exact_times[row_count])),
                        *base_row[2:],
                    ]
                )
                row_count += 1

            if row_count != len(exact_times):
                raise ValueError(
                    f"Landmark CSV has {row_count} rows but ffprobe found "
                    f"{len(exact_times)} displayed frames"
                )

        os.replace(temporary_path, landmark_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return "written_exact_pts", row_count


def persist_exact_driver_pts_for_all_landmarks(
    data_root: str | Path,
    landmark_root: str | Path,
    *,
    driver_video_subfolder: str = driver_landmarks.DEFAULT_DRIVER_VIDEO_SUBFOLDER,
    overwrite: bool = False,
    ffprobe_path: str | Path | None = None,
    ffprobe_timeout_s: float = 300.0,
) -> pd.DataFrame:
    """Persist only exact relative Driver PTS in discovered landmark CSVs."""

    selected_data_root = Path(data_root).expanduser().resolve()
    selected_landmark_root = Path(landmark_root).expanduser().resolve()
    jobs = driver_landmarks.discover_driver_videos(
        data_root=selected_data_root,
        output_root=selected_landmark_root,
        driver_video_subfolder=driver_video_subfolder,
    )
    if not jobs:
        print("No Driver Video files found.")
        return pd.DataFrame(columns=EXACT_PTS_SUMMARY_COLUMNS)

    summaries: list[dict[str, object]] = []
    print(f"Found {len(jobs)} exact Driver PTS persistence job(s).")
    for job_number, job in enumerate(jobs, start=1):
        video_path = Path(job["video_path"]).expanduser().resolve()
        landmark_path = Path(job["output_path"]).expanduser().resolve()
        print(f"[{job_number}/{len(jobs)}] {video_path.name}")
        try:
            status, frames = _persist_exact_pts_one(
                video_path,
                landmark_path,
                overwrite=overwrite,
                ffprobe_path=ffprobe_path,
                ffprobe_timeout_s=ffprobe_timeout_s,
            )
            error = ""
        except Exception as exception:
            status = "error"
            frames = np.nan
            error = repr(exception)
            print(f"  ERROR: {exception}")
        summaries.append(
            {
                "participant": job["participant"],
                "video_path": str(video_path),
                "landmark_csv_path": str(landmark_path),
                "status": status,
                "frames": frames,
                "error": error,
            }
        )
    return pd.DataFrame(summaries, columns=EXACT_PTS_SUMMARY_COLUMNS)


def _write_publication_one(
    landmark_path: Path,
    output_path: Path,
    driver_mapping: pd.Series,
    et_lsl_mapping: pd.Series,
) -> dict[str, object]:
    source_header = _read_header(landmark_path)

    composition = driver_lsl.compose_driver_lsl_mapping(
        driver_mapping,
        et_lsl_mapping,
    )
    driver_intercept = float(driver_mapping["driver_to_et_raw_intercept_s"])
    driver_scale = float(driver_mapping["driver_to_et_raw_scale"])
    lsl_intercept = float(composition["driver_to_lsl_intercept_s"])
    lsl_scale = float(composition["driver_to_lsl_scale"])

    science_columns = driver_landmarks.CSV_COLUMNS[2:]
    science_indices = [source_header.index(column) for column in science_columns]
    frame_index_position = source_header.index("frame_index")
    driver_time_position = source_header.index("driver_video_time_s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    lsl_count = 0
    with (
        landmark_path.open("r", newline="", encoding="utf-8") as source_file,
        output_path.open("w", newline="", encoding="utf-8") as output_file,
    ):
        reader = csv.reader(source_file)
        writer = csv.writer(output_file, lineterminator="\n")
        if next(reader, None) != source_header:
            raise RuntimeError(f"Landmark CSV header changed: {landmark_path}")
        writer.writerow(PUBLICATION_DATA_COLUMNS)

        for source_row in reader:
            if len(source_row) != len(source_header):
                raise ValueError(
                    f"Landmark row {frames} has {len(source_row)} columns; "
                    f"expected {len(source_header)}"
                )
            try:
                frame_index = int(source_row[frame_index_position])
                driver_time = float(source_row[driver_time_position])
            except ValueError as error:
                raise ValueError(
                    f"Landmark row {frames} has invalid frame index or exact PTS"
                ) from error
            if frame_index != frames:
                raise ValueError(
                    f"Landmark row {frames} has frame_index={frame_index}; "
                    f"expected {frames}"
                )
            if not math.isfinite(driver_time):
                raise ValueError(f"Landmark row {frames} has invalid exact PTS")

            et_raw_time = driver_intercept + driver_scale * driver_time
            lsl_time = lsl_intercept + lsl_scale * driver_time
            if not math.isfinite(et_raw_time) or not math.isfinite(lsl_time):
                raise ValueError(f"Landmark row {frames} has non-finite mapped time")
            lsl_count += 1
            science_values = [source_row[index] for index in science_indices]
            writer.writerow(
                [
                    frame_index,
                    _format_time(driver_time),
                    _format_time(et_raw_time),
                    _format_time(lsl_time),
                    *science_values,
                ]
            )
            frames += 1

    if frames == 0:
        raise ValueError(f"Exact-PTS landmark CSV contains no rows: {landmark_path}")
    return {
        "output_path": str(output_path),
        "status": "written_driver_et_lsl" if lsl_count else "written_driver_et",
        "frames": frames,
        "et_raw_time_count": frames,
        "lsl_time_count": lsl_count,
    }


def export_publication_from_synchronized_landmarks(
    landmark_root: str | Path,
    driver_et_alignment_csv: str | Path,
    et_lsl_alignment_csv: str | Path,
    *,
    output_root: str | Path,
    overwrite: bool = False,
    project: str | None = None,
) -> pd.DataFrame:
    """Write compact Driver publication CSVs without reading raw video."""

    selected_landmark_root = Path(landmark_root).expanduser().resolve()
    if not selected_landmark_root.is_dir():
        raise FileNotFoundError(
            f"Driver landmark root does not exist: {selected_landmark_root}"
    )
    selected_output_root = Path(output_root).expanduser().resolve()
    suffix = driver_landmarks.DEFAULT_OUTPUT_SUFFIX
    landmark_paths = sorted(
        path
        for path in selected_landmark_root.iterdir()
        if path.is_file() and path.name.endswith(suffix)
    )
    if not landmark_paths:
        print("No exact-PTS Driver landmark CSV files found.")
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    selected_output_root.mkdir(parents=True, exist_ok=True)
    driver_alignments: pd.DataFrame | None = None
    et_lsl_alignments: pd.DataFrame | None = None
    summaries: list[dict[str, object]] = []
    print(f"Found {len(landmark_paths)} Driver publication job(s).")
    for job_number, landmark_path in enumerate(landmark_paths, start=1):
        driver_mapping: pd.Series | None = None
        et_lsl_mapping: pd.Series | None = None
        error = ""
        try:
            landmark_stem, participant, visit = _landmark_identity(landmark_path)
            output_path = _publication_output_path(
                selected_output_root, landmark_path
            )
            print(f"[{job_number}/{len(landmark_paths)}] {landmark_path.name}")

            if output_path.exists() and not overwrite:
                result = {
                    "output_path": str(output_path),
                    "status": "skipped_existing",
                    "frames": np.nan,
                    "et_raw_time_count": np.nan,
                    "lsl_time_count": np.nan,
                }
            else:
                if driver_alignments is None or et_lsl_alignments is None:
                    loaded_driver_alignments = _read_mapping_table(
                        driver_et_alignment_csv
                    )
                    loaded_et_lsl_alignments = _read_mapping_table(
                        et_lsl_alignment_csv
                    )
                    driver_alignments = loaded_driver_alignments
                    et_lsl_alignments = loaded_et_lsl_alignments
                driver_mapping = _select_driver_mapping(
                    driver_alignments,
                    participant=participant,
                    visit=visit,
                    project=project,
                )
                et_lsl_mapping = _select_et_lsl_mapping(
                    et_lsl_alignments,
                    driver_mapping,
                )
                result = _write_publication_one(
                    landmark_path,
                    output_path,
                    driver_mapping,
                    et_lsl_mapping,
                )
        except Exception as exception:
            landmark_stem = _landmark_stem(landmark_path)
            tokens = landmark_stem.split("_")
            participant = tokens[0] if tokens else ""
            visit = tokens[1] if len(tokens) > 1 else ""
            output_path = _publication_output_path(
                selected_output_root, landmark_path
            )
            result = {
                "output_path": str(output_path),
                "status": "error",
                "frames": np.nan,
                "et_raw_time_count": np.nan,
                "lsl_time_count": np.nan,
            }
            error = repr(exception)
            print(f"  ERROR: {exception}")

        summaries.append(
            {
                "project": _text(
                    driver_mapping.get("project")
                    if driver_mapping is not None
                    else project
                ),
                "participant": participant,
                "visit": visit,
                "segment_id": _text(
                    driver_mapping.get("segment_id")
                    if driver_mapping is not None
                    else ""
                ),
                "video_path": "",
                "landmark_csv_path": str(landmark_path),
                **result,
                "driver_et_mapping_status": (
                    _text(driver_mapping.get("mapping_status"))
                    if driver_mapping is not None
                    else "not_checked"
                ),
                "et_lsl_mapping_status": (
                    _text(et_lsl_mapping.get("mapping_status"))
                    if et_lsl_mapping is not None
                    else "not_checked"
                ),
                "error": error,
            }
        )
    return pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)


def save_driver_timestamp_summary(
    summary: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Write the Driver export summary directly to its configured CSV."""

    frame = pd.DataFrame(summary).copy()
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=SUMMARY_COLUMNS).to_csv(output_path, index=False)
    return output_path


__all__ = [
    "export_publication_from_synchronized_landmarks",
    "persist_exact_driver_pts_for_all_landmarks",
    "save_driver_timestamp_summary",
]
