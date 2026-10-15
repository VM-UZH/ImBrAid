"""Match IDUN XDF chunks to native EEG samples and save the timing summary.

Dependencies: native IDUN CSV files, XDF files containing IDUN, NumPy,
pandas and pyxdf."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd


DEFAULT_CHUNK_SAMPLES = 20
DEFAULT_SAMPLE_RATE_HZ = 250.0
SUMMARY_COLUMNS = [
    "project", "pid", "visit", "repair_status", "usable_relative_time",
    "quality_grade", "review_required", "raw_csv_path", "xdf_path",
    "matched_chunk_ratio", "arrival_offset_reference_p01_s", "warnings",
]


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
                "raw_csv_path": str(raw_candidates[0]) if len(raw_candidates) == 1 else None,
                "xdf_path": str(xdf_candidates[0]) if len(xdf_candidates) == 1 else None,
            })

    columns = [
        "project", "pid", "visit", "discovery_status",
        "raw_csv_path", "xdf_path",
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


def _candidate_starts(raw_values, target, low=0, high=None):
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
        if np.array_equal(raw_values[start:start + len(target)], target, equal_nan=True):
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

    for xdf_start in range(0, len(xdf_values), chunk_samples):
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
                    and np.array_equal(raw_for_match[expected:expected + len(target)],
                                       target, equal_nan=True)):
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
        else:
            raw_end = None

        rows.append({
            "match_status": (
                "unmatched" if raw_start is None else
                "exact_ambiguous" if len(candidates) > 1 else "exact"
            ),
            "raw_start_sample_index": raw_start,
            "raw_end_timestamp_s": (
                float(raw_timestamps[raw_end]) if raw_end is not None else np.nan
            ),
            "xdf_uncorrected_end_timestamp_s": float(xdf_timestamps[xdf_end - 1]),
        })

    return pd.DataFrame(rows)


def _missing_chunks_inside_coverage(chunks, chunk_samples):
    starts = chunks["raw_start_sample_index"].dropna().astype(np.int64).unique()
    if len(starts) < 2:
        return 0
    starts.sort()
    deltas = np.diff(starts)
    return int(np.sum(np.maximum(np.rint(deltas / chunk_samples).astype(int) - 1, 0)))


def _unavailable_summary(
    *, project, pid, visit, status, warning, raw_csv_path=None, xdf_path=None,
):
    return {
        "project": project, "pid": pid, "visit": visit,
        "repair_status": status, "usable_relative_time": False,
        "quality_grade": "D", "review_required": True,
        "raw_csv_path": raw_csv_path, "xdf_path": xdf_path,
        "matched_chunk_ratio": np.nan, "arrival_offset_reference_p01_s": np.nan,
        "warnings": warning,
    }


def _summarise_visit(
    *, project, pid, visit, raw_path, xdf_path,
    raw, xdf_timestamps, chunks, chunk_samples,
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
        arrival_reference = float(np.quantile(
            matched_chunks["xdf_uncorrected_end_timestamp_s"].to_numpy(float)
            - matched_chunks["raw_end_timestamp_s"].to_numpy(float), 0.01,
        ))
    else:
        arrival_reference = np.nan

    nonseq = 0
    if matched_count > 1:
        transitions = np.diff(matched_chunks["raw_start_sample_index"].to_numpy(np.int64))
        nonseq = int(np.sum(transitions != chunk_samples))

    match_ratio = matched_count / total_chunk_count if total_chunk_count else 0.0
    if (raw_timestamp_valid and match_ratio == 1.0 and ambiguous_count == 0
            and duplicate_count == 0 and missing_count == 0):
        quality_grade, usable, review, repair_status = "A", True, False, "repaired"
    elif raw_timestamp_valid and match_ratio >= 0.99:
        quality_grade, usable, review, repair_status = "B", True, True, "repaired_with_review"
    elif raw_timestamp_valid and match_ratio >= 0.95:
        quality_grade, usable, review, repair_status = "C", True, True, "partial_repair"
    else:
        quality_grade, usable, review, repair_status = "D", False, True, "failed"

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
    return {
        "project": project, "pid": pid, "visit": visit,
        "repair_status": repair_status, "usable_relative_time": usable,
        "quality_grade": quality_grade, "review_required": review,
        "raw_csv_path": str(raw_path), "xdf_path": str(xdf_path),
        "matched_chunk_ratio": match_ratio,
        "arrival_offset_reference_p01_s": arrival_reference,
        "warnings": "; ".join(warnings),
    }


def repair_visit(
    job,
    *,
    chunk_samples=DEFAULT_CHUNK_SAMPLES,
    search_radius_s=30.0,
):
    project = str(job["project"])
    pid = str(job["pid"])
    visit = str(job["visit"])
    raw_path = Path(job["raw_csv_path"])
    xdf_path = Path(job["xdf_path"])
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
    return _summarise_visit(
        project=project, pid=pid, visit=visit, raw_path=raw_path, xdf_path=xdf_path,
        raw=raw, xdf_timestamps=xdf_timestamps, chunks=chunks,
        chunk_samples=int(chunk_samples),
    )


def run_idun_repairs(
    jobs,
    output_root,
    *,
    chunk_samples=DEFAULT_CHUNK_SAMPLES,
    search_radius_s=30.0,
):
    output_root = Path(output_root)
    summaries = []
    for job in jobs.to_dict("records"):
        identity = {key: str(job.get(key, "ImBrAid" if key == "project" else ""))
                    for key in ("project", "pid", "visit")}
        if job.get("discovery_status") != "ready":
            summaries.append(_unavailable_summary(
                **identity, status="unavailable",
                warning=f"Discovery status: {job.get('discovery_status')}",
                raw_csv_path=job.get("raw_csv_path"), xdf_path=job.get("xdf_path"),
            ))
            continue
        try:
            summaries.append(repair_visit(
                job, chunk_samples=chunk_samples,
                search_radius_s=search_radius_s,
            ))
        except Exception as error:
            summaries.append(_unavailable_summary(
                **identity, status="failed", warning=f"{type(error).__name__}: {error}",
                raw_csv_path=job.get("raw_csv_path"), xdf_path=job.get("xdf_path"),
            ))
    summary = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "idun_repair_summary.csv", index=False)
    return summary
