"""Reconstruct ImBrAid IDUN EEG timing from the native raw recording.

The ImBrAid IDUN LSL sender pushed 20-sample EEG chunks without supplying the
native sample timestamps.  XDF timestamps therefore describe callback arrival
at the recording computer.  Cloud buffering can make those timestamps overlap
and can return chunks out of acquisition order.

This module matches every XDF chunk to the identical values in the IDUN raw
CSV, restores raw chronological order, and writes an IDUN-relative result.  It
deliberately does not estimate a global IDUN-to-LSL offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd


DEFAULT_CHUNK_SAMPLES = 20
DEFAULT_SAMPLE_RATE_HZ = 250.0
SUMMARY_COLUMNS = [
    "project", "pid", "visit",
    "repair_status", "usable_relative_time", "quality_grade",
    "review_required", "global_lsl_offset_status",
    "global_lsl_time_available", "raw_csv_path", "xdf_path",
    "repaired_csv_path", "chunk_mapping_csv_path",
    "raw_sample_count", "raw_duration_s", "raw_median_dt_s",
    "raw_effective_rate_hz", "raw_max_gap_s", "raw_timestamp_valid",
    "xdf_sample_count", "xdf_uncorrected_duration_s",
    "xdf_nonpositive_dt_count", "xdf_nonpositive_dt_rate",
    "xdf_max_gap_s", "chunk_samples", "total_chunk_count",
    "matched_chunk_count", "matched_chunk_ratio", "unmatched_chunk_count",
    "ambiguous_match_chunk_count",
    "unique_raw_chunk_count", "duplicate_raw_chunk_count",
    "missing_raw_chunk_count_inside_coverage",
    "nonsequential_transition_count", "nonsequential_transition_rate",
    "order_inversion_count", "displaced_chunk_count",
    "displaced_chunk_rate", "max_order_displacement_s",
    "raw_prefix_before_xdf_coverage_s", "raw_suffix_after_xdf_coverage_s",
    "repaired_sample_count", "raw_coverage_ratio",
    "arrival_offset_reference_p01_s", "arrival_offset_relative_median_s",
    "arrival_offset_relative_p95_s", "arrival_offset_relative_p99_s",
    "arrival_offset_relative_max_s", "warnings",
]


ISSUE_COLUMNS = [
    "project", "pid", "visit", "severity", "issue", "detail",
]


@dataclass(frozen=True)
class VisitRepair:
    summary: dict
    chunks: pd.DataFrame
    repaired: pd.DataFrame
    issues: list[dict]


def _normalise_filter(values):
    if values is None:
        return None
    return {str(value).strip() for value in values}


def _first_two_tokens(path: Path):
    tokens = path.stem.split("_")
    return (tokens[0], tokens[1]) if len(tokens) >= 2 else None


def discover_idun_jobs(
    data_root,
    *,
    project="ImBrAid",
    participants=None,
    visits=None,
):
    """Find one native IDUN EEG CSV and one XDF file per participant/visit."""

    root = Path(data_root)
    participant_filter = _normalise_filter(participants)
    visit_filter = _normalise_filter(visits)
    rows = []

    if not root.is_dir():
        raise FileNotFoundError(f"IDUN data root does not exist: {root}")

    for participant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        pid = participant_dir.name
        if participant_filter is not None and pid not in participant_filter:
            continue

        idun_dir = participant_dir / "IDUN"
        raw_by_visit = {}
        if idun_dir.is_dir():
            raw_paths = sorted(
                path for path in idun_dir.iterdir()
                if path.is_file()
                and path.suffix.casefold() == ".csv"
                and "_eeg_" in path.stem.casefold()
            )
            for raw_path in raw_paths:
                identity = _first_two_tokens(raw_path)
                if identity is not None and identity[0] == pid:
                    raw_by_visit.setdefault(identity[1], []).append(raw_path)

        lsl_dir = participant_dir / "LSL"
        xdf_paths = (
            sorted(
                path for path in lsl_dir.iterdir()
                if path.is_file() and path.suffix.casefold() == ".xdf"
            )
            if lsl_dir.is_dir()
            else []
        )
        possible_visits = set(raw_by_visit)
        for xdf_path in xdf_paths:
            identity = _first_two_tokens(xdf_path)
            if identity is not None and identity[0] == pid:
                possible_visits.add(identity[1])

        for visit in sorted(possible_visits):
            if visit_filter is not None and visit not in visit_filter:
                continue
            raw_candidates = raw_by_visit.get(visit, [])
            xdf_candidates = [
                path for path in xdf_paths
                if re.search(rf"(?:^|_){re.escape(visit)}(?:_|$)", path.stem)
            ]
            if len(raw_candidates) == 1 and len(xdf_candidates) == 1:
                status = "ready"
            elif not raw_candidates and not xdf_candidates:
                status = "missing_raw_and_xdf"
            elif not raw_candidates:
                status = "missing_raw"
            elif not xdf_candidates:
                status = "missing_xdf"
            elif len(raw_candidates) > 1:
                status = "ambiguous_raw"
            else:
                status = "ambiguous_xdf"

            rows.append({
                "project": project,
                "pid": pid,
                "visit": visit,
                "discovery_status": status,
                "raw_candidate_count": len(raw_candidates),
                "xdf_candidate_count": len(xdf_candidates),
                "raw_csv_path": str(raw_candidates[0]) if len(raw_candidates) == 1 else None,
                "xdf_path": str(xdf_candidates[0]) if len(xdf_candidates) == 1 else None,
            })

    columns = [
        "project", "pid", "visit", "discovery_status",
        "raw_candidate_count", "xdf_candidate_count", "raw_csv_path", "xdf_path",
    ]
    return pd.DataFrame(rows, columns=columns)


def _read_raw_csv(path: Path):
    raw = pd.read_csv(path, usecols=["timestamp", "ch1"])
    raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce")
    raw["ch1"] = pd.to_numeric(raw["ch1"], errors="coerce")
    if raw.empty:
        raise ValueError(f"IDUN raw EEG CSV is empty: {path}")
    return raw


def _load_uncorrected_idun_stream(path: Path):
    try:
        import pyxdf
    except ImportError as error:
        raise ImportError("pyxdf is required to read IDUN streams from XDF") from error

    streams, header = pyxdf.load_xdf(
        str(path),
        select_streams=[{"name": "IDUN"}],
        synchronize_clocks=False,
        dejitter_timestamps=False,
        verbose=False,
    )
    if len(streams) != 1:
        raise ValueError(f"Expected one IDUN stream in {path}, found {len(streams)}")

    stream = streams[0]
    values = np.asarray(stream["time_series"])
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError(f"Expected a one-channel IDUN stream, found shape {values.shape}")
    timestamps = np.asarray(stream["time_stamps"], dtype=np.float64)
    if len(values) != len(timestamps) or len(values) == 0:
        raise ValueError("IDUN XDF values and timestamps must have equal non-zero length")
    return values[:, 0], timestamps, header


def _header_epoch_s(header):
    try:
        value = header["info"]["datetime"][0]
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _arrays_equal(first, second):
    return np.array_equal(first, second, equal_nan=True)


def _candidate_starts(raw_values, target, low=0, high=None):
    """Return exact target starts, using one finite target value as an index."""

    low = max(0, int(low))
    high = len(raw_values) if high is None else min(len(raw_values), int(high))
    if high <= low or len(target) == 0:
        return []

    finite_positions = np.flatnonzero(np.isfinite(target))
    if finite_positions.size:
        key_offset = int(finite_positions[0])
        key = target[key_offset]
        key_positions = np.flatnonzero(raw_values[low:high] == key) + low
        starts = key_positions - key_offset
    else:
        starts = np.arange(low, high, dtype=int)

    matches = []
    for start in starts:
        start = int(start)
        if start < low or start + len(target) > high:
            continue
        if _arrays_equal(raw_values[start:start + len(target)], target):
            matches.append(start)
    return matches


def _choose_match(candidates, expected_index):
    if not candidates:
        return None
    if expected_index is None:
        return int(candidates[0])
    return int(min(candidates, key=lambda value: abs(value - expected_index)))


def _match_chunks(
    raw_values,
    raw_timestamps,
    xdf_values,
    xdf_timestamps,
    *,
    header_epoch_s=None,
    chunk_samples=DEFAULT_CHUNK_SAMPLES,
    search_radius_samples=7500,
):
    raw_for_match = raw_values.astype(xdf_values.dtype, copy=False)
    rows = []
    previous_match = None

    for chunk_index, xdf_start in enumerate(range(0, len(xdf_values), chunk_samples)):
        xdf_end = min(xdf_start + chunk_samples, len(xdf_values))
        target = xdf_values[xdf_start:xdf_end]

        if previous_match is None:
            expected = None
            if header_epoch_s is not None:
                expected = int(np.searchsorted(raw_timestamps, header_epoch_s, side="left"))
            candidates = _candidate_starts(raw_for_match, target)
        else:
            expected = previous_match + chunk_samples
            if (expected + len(target) <= len(raw_for_match)
                    and _arrays_equal(raw_for_match[expected:expected + len(target)], target)):
                candidates = [expected]
            else:
                candidates = _candidate_starts(
                    raw_for_match,
                    target,
                    expected - search_radius_samples,
                    expected + search_radius_samples + len(target),
                )
                if not candidates:
                    candidates = _candidate_starts(raw_for_match, target)

        raw_start = _choose_match(candidates, expected)
        if raw_start is not None:
            previous_match = raw_start
            raw_end = raw_start + len(target) - 1
            status = "exact" if len(candidates) == 1 else "exact_ambiguous"
        else:
            raw_end = None
            status = "unmatched"

        rows.append({
            "chunk_index": chunk_index,
            "xdf_start_sample_index": xdf_start,
            "xdf_end_sample_index": xdf_end - 1,
            "chunk_sample_count": len(target),
            "match_status": status,
            "match_candidate_count": len(candidates),
            "raw_start_sample_index": raw_start,
            "raw_end_sample_index": raw_end,
            "raw_start_timestamp_s": (
                float(raw_timestamps[raw_start]) if raw_start is not None else np.nan
            ),
            "raw_end_timestamp_s": (
                float(raw_timestamps[raw_end]) if raw_end is not None else np.nan
            ),
            "xdf_uncorrected_start_timestamp_s": float(xdf_timestamps[xdf_start]),
            "xdf_uncorrected_end_timestamp_s": float(xdf_timestamps[xdf_end - 1]),
        })

    chunks = pd.DataFrame(rows)
    _add_chunk_qc_columns(chunks)
    return chunks


def _add_chunk_qc_columns(chunks):
    matched = chunks["raw_start_sample_index"].notna()
    chunks["raw_order_delta_samples"] = np.nan
    chunks["order_inversion"] = False
    chunks["chronological_displacement_chunks"] = np.nan
    chunks["arrival_offset_raw_minus_xdf_clock_s"] = np.nan
    chunks["arrival_offset_relative_to_p01_s"] = np.nan

    if not matched.any():
        return

    matched_index = chunks.index[matched]
    starts = chunks.loc[matched_index, "raw_start_sample_index"].to_numpy(dtype=np.int64)
    deltas = np.diff(starts, prepend=starts[0])
    chunks.loc[matched_index, "raw_order_delta_samples"] = deltas
    chunks.loc[matched_index, "order_inversion"] = deltas < 0

    order = np.argsort(starts, kind="stable")
    chronological_rank = np.empty(len(starts), dtype=np.int64)
    chronological_rank[order] = np.arange(len(starts), dtype=np.int64)
    arrival_rank = np.arange(len(starts), dtype=np.int64)
    chunks.loc[matched_index, "chronological_displacement_chunks"] = (
        chronological_rank - arrival_rank
    )

    raw_end = chunks.loc[matched_index, "raw_end_timestamp_s"].to_numpy(dtype=float)
    xdf_end = chunks.loc[
        matched_index, "xdf_uncorrected_end_timestamp_s"
    ].to_numpy(dtype=float)
    offsets = xdf_end - raw_end
    reference = float(np.quantile(offsets, 0.01))
    chunks.loc[matched_index, "arrival_offset_raw_minus_xdf_clock_s"] = offsets
    chunks.loc[matched_index, "arrival_offset_relative_to_p01_s"] = offsets - reference


def _build_repaired(raw, xdf_timestamps, chunks):
    raw_indices = []
    xdf_indices = []
    chunk_indices = []

    for row in chunks.itertuples(index=False):
        if pd.isna(row.raw_start_sample_index):
            continue
        raw_start = int(row.raw_start_sample_index)
        count = int(row.chunk_sample_count)
        xdf_start = int(row.xdf_start_sample_index)
        raw_indices.append(np.arange(raw_start, raw_start + count, dtype=np.int64))
        xdf_indices.append(np.arange(xdf_start, xdf_start + count, dtype=np.int64))
        chunk_indices.append(np.full(count, int(row.chunk_index), dtype=np.int64))

    if not raw_indices:
        return pd.DataFrame(columns=[
            "raw_sample_index", "idun_raw_timestamp_s", "idun_raw_relative_time_s",
            "idun_lsl_covered_relative_time_s", "eeg_ch1", "source_xdf_sample_index",
            "source_xdf_chunk_index", "xdf_uncorrected_timestamp_s",
        ])

    raw_index = np.concatenate(raw_indices)
    xdf_index = np.concatenate(xdf_indices)
    chunk_index = np.concatenate(chunk_indices)
    repaired = pd.DataFrame({
        "raw_sample_index": raw_index,
        "idun_raw_timestamp_s": raw["timestamp"].to_numpy(dtype=float)[raw_index],
        "eeg_ch1": raw["ch1"].to_numpy(dtype=float)[raw_index],
        "source_xdf_sample_index": xdf_index,
        "source_xdf_chunk_index": chunk_index,
        "xdf_uncorrected_timestamp_s": xdf_timestamps[xdf_index],
    })
    repaired.sort_values(
        ["raw_sample_index", "source_xdf_sample_index"], inplace=True, kind="stable"
    )
    repaired.drop_duplicates("raw_sample_index", keep="first", inplace=True)
    repaired.reset_index(drop=True, inplace=True)
    raw_start_time = float(raw["timestamp"].iloc[0])
    coverage_start = float(repaired["idun_raw_timestamp_s"].iloc[0])
    repaired.insert(
        2,
        "idun_raw_relative_time_s",
        repaired["idun_raw_timestamp_s"] - raw_start_time,
    )
    repaired.insert(
        3,
        "idun_lsl_covered_relative_time_s",
        repaired["idun_raw_timestamp_s"] - coverage_start,
    )
    return repaired


def _missing_chunks_inside_coverage(chunks, chunk_samples):
    starts = chunks["raw_start_sample_index"].dropna().astype(np.int64).unique()
    if len(starts) < 2:
        return 0
    starts.sort()
    deltas = np.diff(starts)
    return int(np.sum(np.maximum(np.rint(deltas / chunk_samples).astype(int) - 1, 0)))


def _issue(project, pid, visit, severity, issue, detail):
    return {
        "project": project,
        "pid": pid,
        "visit": visit,
        "severity": severity,
        "issue": issue,
        "detail": detail,
    }


def _unavailable_summary(
    *, project, pid, visit, output_root, status, warning,
    raw_csv_path=None, xdf_path=None,
):
    row = {column: np.nan for column in SUMMARY_COLUMNS}
    row.update({
        "project": project,
        "pid": pid,
        "visit": visit,
        "repair_status": status,
        "usable_relative_time": False,
        "quality_grade": "D",
        "review_required": True,
        "global_lsl_offset_status": "unresolved",
        "global_lsl_time_available": False,
        "raw_csv_path": raw_csv_path,
        "xdf_path": xdf_path,
        "repaired_csv_path": str(Path(output_root) / f"{pid}_{visit}_idun_repaired.csv"),
        "chunk_mapping_csv_path": str(
            Path(output_root) / "chunks" / f"{pid}_{visit}_idun_chunk_mapping.csv"
        ),
        "warnings": warning,
    })
    return row


def _summarise_visit(
    *, project, pid, visit, raw_path, xdf_path, output_root,
    raw, xdf_timestamps, chunks, repaired, chunk_samples,
):
    raw_timestamps = raw["timestamp"].to_numpy(dtype=float)
    raw_dt = np.diff(raw_timestamps)
    raw_timestamp_valid = bool(
        np.all(np.isfinite(raw_timestamps))
        and len(raw_timestamps) > 1
        and np.all(raw_dt > 0)
    )
    xdf_dt = np.diff(xdf_timestamps)
    matched = chunks["raw_start_sample_index"].notna()
    matched_chunks = chunks.loc[matched]
    total_chunk_count = len(chunks)
    matched_count = int(matched.sum())
    ambiguous_count = int((chunks["match_status"] == "exact_ambiguous").sum())
    unique_count = int(matched_chunks["raw_start_sample_index"].nunique())
    duplicate_count = matched_count - unique_count
    missing_count = _missing_chunks_inside_coverage(chunks, chunk_samples)

    if matched_count:
        raw_start = int(matched_chunks["raw_start_sample_index"].min())
        raw_end = int(matched_chunks["raw_end_sample_index"].max())
        prefix_s = float(raw_timestamps[raw_start] - raw_timestamps[0])
        suffix_s = float(raw_timestamps[-1] - raw_timestamps[raw_end])
        displacement = matched_chunks["chronological_displacement_chunks"].to_numpy(float)
        displaced_count = int(np.sum(displacement != 0))
        arrival_relative = matched_chunks[
            "arrival_offset_relative_to_p01_s"
        ].to_numpy(float)
        arrival_reference = float(np.quantile(
            matched_chunks["arrival_offset_raw_minus_xdf_clock_s"].to_numpy(float),
            0.01,
        ))
    else:
        prefix_s = suffix_s = np.nan
        displacement = np.array([], dtype=float)
        displaced_count = 0
        arrival_relative = np.array([], dtype=float)
        arrival_reference = np.nan

    nonseq = 0
    inversions = 0
    if matched_count > 1:
        starts_in_arrival_order = matched_chunks[
            "raw_start_sample_index"
        ].to_numpy(dtype=np.int64)
        transitions = np.diff(starts_in_arrival_order)
        nonseq = int(np.sum(transitions != chunk_samples))
        inversions = int(np.sum(transitions < 0))

    match_ratio = matched_count / total_chunk_count if total_chunk_count else 0.0
    if (raw_timestamp_valid and match_ratio == 1.0 and ambiguous_count == 0
            and duplicate_count == 0 and missing_count == 0):
        quality_grade = "A"
        usable = True
        review = False
        repair_status = "repaired"
    elif raw_timestamp_valid and match_ratio >= 0.99:
        quality_grade = "B"
        usable = True
        review = True
        repair_status = "repaired_with_review"
    elif raw_timestamp_valid and match_ratio >= 0.95:
        quality_grade = "C"
        usable = True
        review = True
        repair_status = "partial_repair"
    else:
        quality_grade = "D"
        usable = False
        review = True
        repair_status = "failed"

    warnings = []
    if nonseq:
        warnings.append(f"{nonseq} chunk transitions were not in raw acquisition order")
    if ambiguous_count:
        warnings.append(f"{ambiguous_count} chunks had more than one exact raw candidate")
    nonpositive = int(np.sum(xdf_dt <= 0))
    if nonpositive:
        warnings.append(f"{nonpositive} uncorrected XDF sample intervals were non-positive")
    if len(xdf_dt) and float(np.max(xdf_dt)) > 1.0:
        warnings.append(f"maximum uncorrected XDF gap was {float(np.max(xdf_dt)):.3f} s")
    warnings.append("global IDUN-to-LSL offset is unresolved")

    repaired_path = output_root / f"{pid}_{visit}_idun_repaired.csv"
    chunks_path = output_root / "chunks" / f"{pid}_{visit}_idun_chunk_mapping.csv"

    def quantile(values, probability):
        return float(np.quantile(values, probability)) if len(values) else np.nan

    summary = {
        "project": project,
        "pid": pid,
        "visit": visit,
        "repair_status": repair_status,
        "usable_relative_time": usable,
        "quality_grade": quality_grade,
        "review_required": review,
        "global_lsl_offset_status": "unresolved",
        "global_lsl_time_available": False,
        "raw_csv_path": str(raw_path),
        "xdf_path": str(xdf_path),
        "repaired_csv_path": str(repaired_path),
        "chunk_mapping_csv_path": str(chunks_path),
        "raw_sample_count": len(raw),
        "raw_duration_s": float(raw_timestamps[-1] - raw_timestamps[0]),
        "raw_median_dt_s": quantile(raw_dt, 0.5),
        "raw_effective_rate_hz": 1.0 / quantile(raw_dt, 0.5) if len(raw_dt) else np.nan,
        "raw_max_gap_s": float(np.max(raw_dt)) if len(raw_dt) else np.nan,
        "raw_timestamp_valid": raw_timestamp_valid,
        "xdf_sample_count": len(xdf_timestamps),
        "xdf_uncorrected_duration_s": float(xdf_timestamps[-1] - xdf_timestamps[0]),
        "xdf_nonpositive_dt_count": nonpositive,
        "xdf_nonpositive_dt_rate": nonpositive / len(xdf_dt) if len(xdf_dt) else np.nan,
        "xdf_max_gap_s": float(np.max(xdf_dt)) if len(xdf_dt) else np.nan,
        "chunk_samples": chunk_samples,
        "total_chunk_count": total_chunk_count,
        "matched_chunk_count": matched_count,
        "matched_chunk_ratio": match_ratio,
        "unmatched_chunk_count": total_chunk_count - matched_count,
        "ambiguous_match_chunk_count": ambiguous_count,
        "unique_raw_chunk_count": unique_count,
        "duplicate_raw_chunk_count": duplicate_count,
        "missing_raw_chunk_count_inside_coverage": missing_count,
        "nonsequential_transition_count": nonseq,
        "nonsequential_transition_rate": nonseq / (matched_count - 1) if matched_count > 1 else 0.0,
        "order_inversion_count": inversions,
        "displaced_chunk_count": displaced_count,
        "displaced_chunk_rate": displaced_count / matched_count if matched_count else np.nan,
        "max_order_displacement_s": (
            float(np.max(np.abs(displacement))) * chunk_samples / DEFAULT_SAMPLE_RATE_HZ
            if len(displacement) else np.nan
        ),
        "raw_prefix_before_xdf_coverage_s": prefix_s,
        "raw_suffix_after_xdf_coverage_s": suffix_s,
        "repaired_sample_count": len(repaired),
        "raw_coverage_ratio": len(repaired) / len(raw),
        "arrival_offset_reference_p01_s": arrival_reference,
        "arrival_offset_relative_median_s": quantile(arrival_relative, 0.5),
        "arrival_offset_relative_p95_s": quantile(arrival_relative, 0.95),
        "arrival_offset_relative_p99_s": quantile(arrival_relative, 0.99),
        "arrival_offset_relative_max_s": float(np.max(arrival_relative)) if len(arrival_relative) else np.nan,
        "warnings": "; ".join(warnings),
    }

    issues = []
    if not raw_timestamp_valid:
        issues.append(_issue(project, pid, visit, "error", "invalid_raw_timestamps",
                             "IDUN raw timestamps are not finite and strictly increasing"))
    if matched_count < total_chunk_count:
        issues.append(_issue(project, pid, visit, "error", "unmatched_xdf_chunks",
                             f"{total_chunk_count - matched_count} of {total_chunk_count} chunks were unmatched"))
    if ambiguous_count:
        issues.append(_issue(project, pid, visit, "warning", "ambiguous_chunk_matches",
                             f"{ambiguous_count} chunks had more than one exact raw candidate"))
    if duplicate_count:
        issues.append(_issue(project, pid, visit, "warning", "duplicate_raw_chunks",
                             f"{duplicate_count} XDF chunks mapped to raw chunks already represented"))
    if missing_count:
        issues.append(_issue(project, pid, visit, "warning", "missing_raw_chunks",
                             f"{missing_count} raw chunks were absent inside XDF coverage"))
    if nonseq:
        issues.append(_issue(project, pid, visit, "warning", "xdf_chunk_reordering",
                             f"{nonseq} adjacent XDF chunk transitions were not chronological"))
    if nonpositive:
        issues.append(_issue(project, pid, visit, "warning", "nonmonotonic_xdf_timestamps",
                             f"{nonpositive} uncorrected XDF sample intervals were non-positive"))
    issues.append(_issue(project, pid, visit, "info", "global_lsl_offset_unresolved",
                         "The repaired file uses IDUN raw time and contains no global LSL time"))
    return summary, issues


def repair_visit(
    job,
    output_root,
    *,
    chunk_samples=DEFAULT_CHUNK_SAMPLES,
    search_radius_s=30.0,
):
    """Repair one ready discovery row without estimating a global LSL offset."""

    project = str(job["project"])
    pid = str(job["pid"])
    visit = str(job["visit"])
    raw_path = Path(job["raw_csv_path"])
    xdf_path = Path(job["xdf_path"])
    output_root = Path(output_root)

    raw = _read_raw_csv(raw_path)
    xdf_values, xdf_timestamps, header = _load_uncorrected_idun_stream(xdf_path)
    chunks = _match_chunks(
        raw["ch1"].to_numpy(dtype=float),
        raw["timestamp"].to_numpy(dtype=float),
        xdf_values,
        xdf_timestamps,
        header_epoch_s=_header_epoch_s(header),
        chunk_samples=int(chunk_samples),
        search_radius_samples=int(round(search_radius_s * DEFAULT_SAMPLE_RATE_HZ)),
    )
    repaired = _build_repaired(raw, xdf_timestamps, chunks)
    summary, issues = _summarise_visit(
        project=project,
        pid=pid,
        visit=visit,
        raw_path=raw_path,
        xdf_path=xdf_path,
        output_root=output_root,
        raw=raw,
        xdf_timestamps=xdf_timestamps,
        chunks=chunks,
        repaired=repaired,
        chunk_samples=int(chunk_samples),
    )
    chunks.insert(0, "visit", visit)
    chunks.insert(0, "pid", pid)
    chunks.insert(0, "project", project)
    return VisitRepair(summary, chunks, repaired, issues)


def run_idun_repairs(
    jobs,
    output_root,
    *,
    chunk_samples=DEFAULT_CHUNK_SAMPLES,
    search_radius_s=30.0,
):
    """Repair every ready job and save per-visit data plus one run manifest."""

    output_root = Path(output_root)
    summaries = []
    issues = []

    for job in jobs.to_dict("records"):
        project = str(job.get("project", "ImBrAid"))
        pid = str(job.get("pid", ""))
        visit = str(job.get("visit", ""))
        if job.get("discovery_status") != "ready":
            detail = f"Discovery status: {job.get('discovery_status')}"
            summaries.append(_unavailable_summary(
                project=project,
                pid=pid,
                visit=visit,
                output_root=output_root,
                status="unavailable",
                warning=detail,
                raw_csv_path=job.get("raw_csv_path"),
                xdf_path=job.get("xdf_path"),
            ))
            issues.append(_issue(project, pid, visit, "error", "source_discovery_failed", detail))
            continue
        try:
            result = repair_visit(
                job,
                output_root,
                chunk_samples=chunk_samples,
                search_radius_s=search_radius_s,
            )
            summary = result.summary
            Path(summary["repaired_csv_path"]).parent.mkdir(parents=True, exist_ok=True)
            Path(summary["chunk_mapping_csv_path"]).parent.mkdir(parents=True, exist_ok=True)
            result.repaired.to_csv(summary["repaired_csv_path"], index=False)
            result.chunks.to_csv(summary["chunk_mapping_csv_path"], index=False)
            summaries.append(summary)
            issues.extend(result.issues)
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            summaries.append(_unavailable_summary(
                project=project,
                pid=pid,
                visit=visit,
                output_root=output_root,
                status="failed",
                warning=detail,
                raw_csv_path=job.get("raw_csv_path"),
                xdf_path=job.get("xdf_path"),
            ))
            issues.append(_issue(
                project, pid, visit, "error", "repair_failed",
                detail,
            ))

    summary_frame = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(output_root / "idun_repair_summary.csv", index=False)
    issue_frame.to_csv(output_root / "idun_repair_issues.csv", index=False)
    return summary_frame, issue_frame
