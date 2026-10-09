"""Content alignment between Tobii ET raw and the eyeball-video pupil trace.

The two inputs are sampled independently and use different units.  ET raw
contains pupil diameter in millimetres on the Tobii recording clock; the
eyeball pipeline contains pupil contour area in pixels on the video clock.
Rows are therefore never paired.  We compare ET diameter with the square root
of image area (a length-like quantity), normalise each local window, estimate
one lag per eye, and fit

    t_eyeball = intercept_s + scale * t_et_raw

Missing pupil samples are availability masks only; they are not treated as
blink events or filled with zero.  The amplitude regression is diagnostic and
does not determine the clock mapping.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from helpers.lsl_et_sync import et_loaders
from helpers.lsl_et_sync import offset as offset_core


GROUP_COLUMNS = ["project", "pid", "visit"]
PUPIL_GROUPS = {
    "left-pupil": ["left-pupil"],
    "right-pupil": ["right-pupil"],
}
ACCEPTED_CONFIDENCE = frozenset({"high", "medium"})

DEFAULT_WINDOW_S = 90.0
DEFAULT_STEP_S = 45.0
DEFAULT_LOCAL_RESIDUAL_SEARCH_RADIUS_S = 15.0
DEFAULT_MAX_GAP_S = 0.25
DEFAULT_SAMPLE_HZ = 20.0
DEFAULT_COARSE_SAMPLE_HZ = 5.0
DEFAULT_MIN_CORRELATION = 0.25
DEFAULT_MIN_PEAK_MARGIN = 0.02
DEFAULT_MIN_GOOD_WINDOWS = 3
DEFAULT_REVIEW_DRIFT_PPM = 2_000.0
DEFAULT_MAX_DRIFT_PPM = 10_000.0
DEFAULT_REVIEW_RESIDUAL_P95_S = 0.25
DEFAULT_MAX_RESIDUAL_P95_S = 0.5
DEFAULT_EYE_OFFSET_DIFFERENCE_S = 0.25
DEFAULT_EYE_DRIFT_DIFFERENCE_PPM = 1_000.0
DEFAULT_MIN_AMPLITUDE_R = 0.15

DISCOVERY_COLUMNS = [
    *GROUP_COLUMNS, "et_recording_path", "pupil_csv_path",
    "et_candidate_count", "pupil_csv_candidate_count", "discovery_status",
    "et_candidates", "pupil_csv_candidates",
]

ALIGNMENT_COLUMNS = [
    *GROUP_COLUMNS, "processing_status", "mapping_status", "mapping_source",
    "discovery_status", "et_candidate_count", "pupil_csv_candidate_count",
    "usable", "quality_grade", "review_required", "mapping_reason",
    "time_equation", "amplitude_rule", "intercept_s", "scale", "drift_ppm",
    "reference_et_s", "offset_at_reference_s", "coarse_prior_s",
    "coarse_prior_source", "n_coarse_priors", "n_windows", "n_good_windows",
    "fit_inliers", "n_eyes", "eyes", "median_peak_correlation",
    "median_peak_margin", "median_eye_spread_s", "fit_residual_mad_s",
    "fit_residual_p95_s", "good_window_ratio", "temporal_coverage_ratio",
    "left_fit_status", "left_intercept_s", "left_scale", "left_drift_ppm",
    "left_good_windows", "left_median_correlation", "left_residual_p95_s",
    "right_fit_status", "right_intercept_s", "right_scale", "right_drift_ppm",
    "right_good_windows", "right_median_correlation", "right_residual_p95_s",
    "eye_offset_difference_s", "eye_drift_difference_ppm",
    "left_amplitude_samples", "left_sqrt_area_intercept",
    "left_sqrt_area_per_mm", "left_amplitude_r", "left_amplitude_r2",
    "right_amplitude_samples", "right_sqrt_area_intercept",
    "right_sqrt_area_per_mm", "right_amplitude_r", "right_amplitude_r2",
    "et_start_s", "et_end_s", "et_span_s", "et_rows",
    "et_duplicate_timestamps", "et_nonincreasing_timestamps", "et_max_gap_s",
    "eyeball_start_s", "eyeball_end_s", "eyeball_span_s", "eyeball_rows",
    "eyeball_duplicate_timestamps", "eyeball_nonincreasing_timestamps",
    "eyeball_max_gap_s", "et_left_valid_ratio", "et_right_valid_ratio",
    "eyeball_left_valid_ratio", "eyeball_right_valid_ratio",
    "et_recording_path", "pupil_csv_path", "settings_json", "warnings",
]

WINDOW_COLUMNS = [
    *GROUP_COLUMNS, "fit_family",
    "center_et_s", "eye", "offset_s", "peak_correlation", "peak_margin",
    "overlap_s", "sample_hz", "distance_from_prior_s", "at_search_boundary",
    "passes_peak_qc",
]

ISSUE_COLUMNS = [
    *GROUP_COLUMNS,
    "severity", "code", "message",
]


def _normalise_filter(values):
    return None if values is None else {str(value) for value in values}


def _pid_visit(name):
    parts = str(name).split("_")
    return (parts[0], parts[1]) if len(parts) >= 2 else None


def discover_et_eyeball_jobs(
    data_root,
    pupil_csv_root,
    project,
    participants=None,
    visits=None,
    et_subfolder="ET",
):
    """Discover exact ET-recording/pupil-CSV pairs without choosing ambiguities."""
    root = Path(data_root)
    pupil_root = Path(pupil_csv_root)
    participant_filter = _normalise_filter(participants)
    visit_filter = _normalise_filter(visits)
    et_by_key, pupil_by_key = {}, {}

    if root.is_dir():
        for participant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            pid = participant_dir.name
            if participant_filter is not None and pid not in participant_filter:
                continue
            et_dir = participant_dir / et_subfolder
            if not et_dir.is_dir():
                continue
            for recording in sorted(path for path in et_dir.iterdir() if path.is_dir()):
                if not (recording / et_loaders.GAZE_FILE).is_file():
                    continue
                key = _pid_visit(recording.name)
                if key and key[0] == pid:
                    et_by_key.setdefault(key, []).append(recording.resolve())

    if pupil_root.is_dir():
        # The configured directory is the canonical pupil-output directory.
        for csv_path in sorted(pupil_root.glob("*_pupil_sizes.csv")):
            key = _pid_visit(csv_path.name)
            if not key:
                continue
            if participant_filter is not None and key[0] not in participant_filter:
                continue
            pupil_by_key.setdefault(key, []).append(csv_path.resolve())

    rows = []
    for pid, visit in sorted(set(et_by_key) | set(pupil_by_key)):
        if visit_filter is not None and visit not in visit_filter:
            continue
        et_candidates = sorted(set(et_by_key.get((pid, visit), [])))
        pupil_candidates = sorted(set(pupil_by_key.get((pid, visit), [])))
        if len(et_candidates) == 1 and len(pupil_candidates) == 1:
            status = "ready"
        elif not et_candidates:
            status = "missing_et_raw"
        elif not pupil_candidates:
            status = "missing_pupil_csv"
        elif len(et_candidates) > 1:
            status = "ambiguous_et_raw"
        else:
            status = "ambiguous_pupil_csv"
        rows.append({
            "project": str(project), "pid": pid, "visit": visit,
            "et_recording_path": str(et_candidates[0]) if len(et_candidates) == 1 else "",
            "pupil_csv_path": str(pupil_candidates[0]) if len(pupil_candidates) == 1 else "",
            "et_candidate_count": len(et_candidates),
            "pupil_csv_candidate_count": len(pupil_candidates),
            "discovery_status": status,
            "et_candidates": " | ".join(map(str, et_candidates)),
            "pupil_csv_candidates": " | ".join(map(str, pupil_candidates)),
        })
    return pd.DataFrame(rows, columns=DISCOVERY_COLUMNS)


def _timestamp_audit(values):
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    finite = numeric[np.isfinite(numeric)]
    diffs = np.diff(finite)
    positive = diffs[diffs > 0]
    return {
        "start_s": float(finite[0]) if len(finite) else np.nan,
        "end_s": float(finite[-1]) if len(finite) else np.nan,
        "span_s": float(finite[-1] - finite[0]) if len(finite) else np.nan,
        "rows": int(len(numeric)),
        "duplicate_timestamps": int(np.sum(diffs == 0)),
        "nonincreasing_timestamps": int(np.sum(diffs <= 0)),
        "max_gap_s": float(np.max(positive)) if len(positive) else np.nan,
    }


def _load_sources(recording_path, pupil_csv_path):
    gaze = et_loaders.load_et_recording(recording_path)
    if gaze is None or gaze.empty:
        raise ValueError("ET raw gazedata is unavailable or empty")
    required_gaze = {"timestamp", "left-pupil", "right-pupil"}
    missing = required_gaze - set(gaze.columns)
    if missing:
        raise ValueError(f"ET raw is missing columns: {sorted(missing)}")

    required_eye = {
        "Time_ms", "Left_Pupil_Area", "Right_Pupil_Area",
        "Left_Confidence", "Right_Confidence",
    }
    eye = pd.read_csv(pupil_csv_path, usecols=sorted(required_eye))
    if eye.empty:
        raise ValueError("pupil CSV is empty")

    et_time = pd.to_numeric(gaze["timestamp"], errors="coerce").to_numpy(float)
    eye_time = pd.to_numeric(eye["Time_ms"], errors="coerce").to_numpy(float) / 1000.0
    et_audit = _timestamp_audit(et_time)
    eye_audit = _timestamp_audit(eye_time)

    et_prepared = pd.DataFrame({"timestamp": et_time})
    eye_prepared = pd.DataFrame(index=pd.to_datetime(eye_time, unit="s", utc=True))
    valid_ratios = {}
    for side, et_column, area_column, confidence_column in (
        ("left", "left-pupil", "Left_Pupil_Area", "Left_Confidence"),
        ("right", "right-pupil", "Right_Pupil_Area", "Right_Confidence"),
    ):
        diameter = pd.to_numeric(gaze[et_column], errors="coerce").to_numpy(float)
        diameter[~np.isfinite(diameter) | (diameter <= 0)] = np.nan
        area = pd.to_numeric(eye[area_column], errors="coerce").to_numpy(float)
        confidence = eye[confidence_column].astype(str).str.lower().to_numpy()
        accepted = np.isin(confidence, list(ACCEPTED_CONFIDENCE))
        area[~np.isfinite(area) | (area <= 0) | ~accepted] = np.nan
        et_prepared[et_column] = diameter
        eye_prepared[et_column] = np.sqrt(area)
        valid_ratios[f"et_{side}_valid_ratio"] = float(np.isfinite(diameter).mean())
        valid_ratios[f"eyeball_{side}_valid_ratio"] = float(np.isfinite(area).mean())

    et_prepared = et_prepared[np.isfinite(et_prepared["timestamp"])].copy()
    eye_prepared = eye_prepared[~eye_prepared.index.isna()].copy()
    return et_prepared, eye_prepared, et_audit, eye_audit, valid_ratios


def _deduplicate_priors(priors, tolerance_s=0.25, limit=8):
    output = []
    for value, source in priors:
        if not np.isfinite(value):
            continue
        if any(abs(float(value) - previous[0]) <= tolerance_s for previous in output):
            continue
        output.append((float(value), str(source)))
        if len(output) >= limit:
            break
    return output


def _coarse_priors(et_frame, eye_frame, et_audit, eye_audit, sample_hz):
    priors = []
    try:
        candidates = offset_core.offset_candidates(
            et_frame, eye_frame, sample_hz=float(sample_hz), verbose=False)
    except (ValueError, TypeError, FloatingPointError):
        candidates = []
    for rank, candidate in enumerate(candidates[:5]):
        priors.append((candidate[0], f"content_candidate_{rank + 1}"))
    priors.extend([
        (eye_audit["end_s"] - et_audit["end_s"], "end_time_difference"),
        (eye_audit["start_s"] - et_audit["start_s"], "start_time_difference"),
    ])
    return _deduplicate_priors(priors)


def _fit_rank(fit):
    if not fit:
        return (-1, -1, -np.inf, -np.inf, -np.inf)
    return (
        int(fit.get("n_good_windows", 0)), int(fit.get("n_groups", 0)),
        float(fit.get("temporal_coverage_ratio", 0.0)),
        float(fit.get("median_peak_correlation", -np.inf)),
        -float(fit.get("fit_residual_mad_s", np.inf)),
    )


def _best_fit(et_frame, eye_frame, groups, priors, settings):
    active_columns = [feature for features in groups.values() for feature in features]
    et_subset = et_frame[["timestamp"] + active_columns]
    eye_subset = eye_frame[active_columns]
    fits = []
    for prior, source in priors:
        fit = offset_core.estimate_windowed_alignment(
            et_subset, eye_subset, initial_offset_s=prior,
            window_s=settings["window_s"], step_s=settings["step_s"],
            search_radius_s=settings["local_residual_search_radius_s"],
            max_gap_s=settings["max_gap_s"],
            min_peak_correlation=settings["min_peak_correlation"],
            min_peak_margin=settings["min_peak_margin"], verbose=False,
            sample_hz_by_group={name: settings["sample_hz"] for name in groups},
            max_abs_drift_ppm=None, min_groups_per_window=1,
        )
        if fit is not None:
            fit = dict(fit)
            fit["coarse_prior_source"] = source
            fits.append(fit)
    return max(fits, key=_fit_rank) if fits else None


def _fit_status(fit, settings):
    if not fit or int(fit.get("n_good_windows", 0)) == 0:
        return "unavailable", ["no local pupil window passed correlation QC"]
    count = int(fit.get("n_good_windows", 0))
    if count < int(settings["min_good_windows"]):
        return "offset_only", [f"only {count} good local window(s); drift is not estimated"]
    drift = abs(float(fit.get("drift_ppm", np.inf)))
    residual = float(fit.get("fit_residual_p95_s", np.inf))
    if drift > settings["max_drift_ppm"]:
        return "invalid", [f"absolute clock drift {drift:.1f} ppm exceeds hard limit"]
    if residual > settings["max_residual_p95_s"]:
        return "invalid", [f"local lag residual P95 {residual:.3f} s exceeds hard limit"]
    reasons = []
    if drift > settings["review_drift_ppm"]:
        reasons.append(f"absolute clock drift {drift:.1f} ppm requires review")
    if residual > settings["review_residual_p95_s"]:
        reasons.append(f"local lag residual P95 {residual:.3f} s requires review")
    if float(fit.get("good_window_ratio", 0.0)) < 0.10:
        reasons.append("fewer than 10% of candidate windows passed QC")
    if float(fit.get("temporal_coverage_ratio", 0.0)) < 0.25:
        reasons.append("good windows cover less than 25% of the shared time range")
    return ("review", reasons) if reasons else ("good", [])


def _mapping_at(fit, et_time):
    return float(fit["intercept_s"]) + (float(fit["scale"]) - 1.0) * float(et_time)


def _fit_summary(side, fit, status):
    return {
        f"{side}_fit_status": status,
        f"{side}_intercept_s": fit.get("intercept_s", np.nan) if fit else np.nan,
        f"{side}_scale": fit.get("scale", np.nan) if fit else np.nan,
        f"{side}_drift_ppm": fit.get("drift_ppm", np.nan) if fit else np.nan,
        f"{side}_good_windows": fit.get("n_good_windows", 0) if fit else 0,
        f"{side}_median_correlation": (
            fit.get("median_peak_correlation", np.nan) if fit else np.nan),
        f"{side}_residual_p95_s": fit.get("fit_residual_p95_s", np.nan) if fit else np.nan,
    }


def _window_rows(job, family, fit, settings):
    if not fit:
        return []
    rows = []
    for window in fit.get("group_windows", []):
        rows.append({
            "project": job.get("project", ""), "pid": job.get("pid", ""),
            "visit": job.get("visit", ""), "fit_family": family,
            "center_et_s": window.get("center_et_s"), "eye": window.get("group"),
            "offset_s": window.get("offset_s"),
            "peak_correlation": window.get("peak_correlation"),
            "peak_margin": window.get("peak_margin"), "overlap_s": window.get("overlap_s"),
            "sample_hz": window.get("sample_hz"),
            "distance_from_prior_s": window.get("distance_from_prior_s"),
            "at_search_boundary": bool(window.get("at_search_boundary", False)),
            "passes_peak_qc": bool(
                window.get("peak_correlation", -np.inf) >= settings["min_peak_correlation"]
                and window.get("peak_margin", -np.inf) >= settings["min_peak_margin"]
                and not window.get("at_search_boundary", False)),
        })
    return rows


def _interpolate_with_gap_mask(times, values, grid, max_gap_s):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(times) & np.isfinite(values)
    times, values = times[valid], values[valid]
    if len(times) < 2:
        return np.full(len(grid), np.nan)
    order = np.argsort(times)
    times, values = times[order], values[order]
    unique, index = np.unique(times[::-1], return_index=True)
    keep = len(times) - 1 - index
    order = np.argsort(unique)
    times, values = unique[order], values[keep[order]]
    right = np.searchsorted(times, grid, side="left")
    inside = (right > 0) & (right < len(times))
    left_index = np.clip(right - 1, 0, len(times) - 1)
    right_index = np.clip(right, 0, len(times) - 1)
    allowed = inside & ((times[right_index] - times[left_index]) <= max_gap_s)
    output = np.interp(grid, times, values)
    output[~allowed] = np.nan
    return output


def _amplitude_regression(et_frame, eye_frame, side, intercept, scale,
                          max_gap_s, sample_hz=10.0):
    column = f"{side}-pupil"
    et_times = et_frame["timestamp"].to_numpy(float) * scale + intercept
    et_values = et_frame[column].to_numpy(float)
    eye_times = eye_frame.index.view("int64").astype(float) / 1e9
    eye_values = eye_frame[column].to_numpy(float)
    start = max(np.nanmin(et_times), np.nanmin(eye_times))
    end = min(np.nanmax(et_times), np.nanmax(eye_times))
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return {"samples": 0, "intercept": np.nan, "slope": np.nan,
                "r": np.nan, "r2": np.nan}
    grid = np.arange(start, end, 1.0 / float(sample_hz))
    diameter = _interpolate_with_gap_mask(et_times, et_values, grid, max_gap_s)
    sqrt_area = _interpolate_with_gap_mask(eye_times, eye_values, grid, max_gap_s)
    valid = np.isfinite(diameter) & np.isfinite(sqrt_area)
    if valid.sum() < 100:
        return {"samples": int(valid.sum()), "intercept": np.nan, "slope": np.nan,
                "r": np.nan, "r2": np.nan}
    x, y = diameter[valid], sqrt_area[valid]
    slope, amplitude_intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
    return {
        "samples": int(valid.sum()), "intercept": float(amplitude_intercept),
        "slope": float(slope), "r": r, "r2": float(r * r) if np.isfinite(r) else np.nan,
    }


def build_et_eyeball_alignments(
    jobs, *, window_s=DEFAULT_WINDOW_S, step_s=DEFAULT_STEP_S,
    local_residual_search_radius_s=DEFAULT_LOCAL_RESIDUAL_SEARCH_RADIUS_S,
    max_gap_s=DEFAULT_MAX_GAP_S, sample_hz=DEFAULT_SAMPLE_HZ,
    coarse_sample_hz=DEFAULT_COARSE_SAMPLE_HZ,
    min_peak_correlation=DEFAULT_MIN_CORRELATION,
    min_peak_margin=DEFAULT_MIN_PEAK_MARGIN,
    min_good_windows=DEFAULT_MIN_GOOD_WINDOWS,
    review_drift_ppm=DEFAULT_REVIEW_DRIFT_PPM,
    max_drift_ppm=DEFAULT_MAX_DRIFT_PPM,
    review_residual_p95_s=DEFAULT_REVIEW_RESIDUAL_P95_S,
    max_residual_p95_s=DEFAULT_MAX_RESIDUAL_P95_S,
    eye_offset_difference_s=DEFAULT_EYE_OFFSET_DIFFERENCE_S,
    eye_drift_difference_ppm=DEFAULT_EYE_DRIFT_DIFFERENCE_PPM,
    min_amplitude_r=DEFAULT_MIN_AMPLITUDE_R,
):
    """Build ET-raw -> Eyeball mappings, local-window QC and explicit issues."""
    settings = {
        "window_s": float(window_s), "step_s": float(step_s),
        "local_residual_search_radius_s": float(local_residual_search_radius_s),
        "max_gap_s": float(max_gap_s), "sample_hz": float(sample_hz),
        "coarse_sample_hz": float(coarse_sample_hz),
        "min_peak_correlation": float(min_peak_correlation),
        "min_peak_margin": float(min_peak_margin),
        "min_good_windows": int(min_good_windows),
        "review_drift_ppm": float(review_drift_ppm),
        "max_drift_ppm": float(max_drift_ppm),
        "review_residual_p95_s": float(review_residual_p95_s),
        "max_residual_p95_s": float(max_residual_p95_s),
        "eye_offset_difference_s": float(eye_offset_difference_s),
        "eye_drift_difference_ppm": float(eye_drift_difference_ppm),
        "min_amplitude_r": float(min_amplitude_r),
    }
    positive = ("window_s", "step_s", "local_residual_search_radius_s", "max_gap_s",
                "sample_hz", "coarse_sample_hz")
    if any(settings[name] <= 0 for name in positive):
        raise ValueError("window, step, search radius, gap and sample rates must be positive")
    if settings["min_good_windows"] < 1:
        raise ValueError("min_good_windows must be at least 1")
    if not 0 <= settings["review_drift_ppm"] <= settings["max_drift_ppm"]:
        raise ValueError("drift thresholds must satisfy 0 <= review <= max")
    if not 0 <= settings["review_residual_p95_s"] <= settings["max_residual_p95_s"]:
        raise ValueError("residual thresholds must satisfy 0 <= review <= max")
    if not -1 <= settings["min_amplitude_r"] <= 1:
        raise ValueError("min_amplitude_r must be between -1 and 1")

    mappings, windows, issues = [], [], []

    def add_issue(job, severity, code, message):
        issues.append({
            "project": job.get("project", ""),
            "pid": job.get("pid", ""), "visit": job.get("visit", ""),
            "severity": severity, "code": code, "message": str(message),
        })

    for _, job_series in jobs.iterrows():
        job = job_series.to_dict()
        base = {column: np.nan for column in ALIGNMENT_COLUMNS}
        base.update({
            "project": job.get("project", ""), "pid": job.get("pid", ""),
            "visit": job.get("visit", ""), "processing_status": "complete",
            "mapping_status": "unavailable", "mapping_source": "none",
            "discovery_status": job.get("discovery_status", ""),
            "et_candidate_count": job.get("et_candidate_count", np.nan),
            "pupil_csv_candidate_count": job.get("pupil_csv_candidate_count", np.nan),
            "usable": False, "quality_grade": "D", "review_required": True,
            "time_equation": "eyeball_s = intercept_s + scale * et_raw_s",
            "amplitude_rule": "sqrt(pixel_area) = alpha + beta * pupil_diameter_mm",
            "settings_json": json.dumps(settings, sort_keys=True), "warnings": "",
        })
        if job.get("discovery_status") != "ready":
            reason = str(job.get("discovery_status", "not ready"))
            base["mapping_reason"] = reason
            add_issue(job, "error", "discovery_not_ready", reason)
            mappings.append(base)
            continue
        try:
            recording_path = Path(job["et_recording_path"])
            pupil_path = Path(job["pupil_csv_path"])
            (et_frame, eye_frame, et_audit, eye_audit,
             valid_ratios) = _load_sources(recording_path, pupil_path)
            for prefix, audit in (("et", et_audit), ("eyeball", eye_audit)):
                for name, value in audit.items():
                    base[f"{prefix}_{name}"] = value
            base.update(valid_ratios)

            base.update({
                "et_recording_path": str(recording_path),
                "pupil_csv_path": str(pupil_path),
            })
            hard_timestamp_error = bool(
                et_audit["nonincreasing_timestamps"]
                or eye_audit["nonincreasing_timestamps"])
            if hard_timestamp_error:
                base.update({"mapping_status": "invalid_timestamps",
                             "mapping_reason": "source timestamps are not strictly increasing"})
                add_issue(job, "error", "nonincreasing_timestamps", base["mapping_reason"])
                mappings.append(base)
                continue

            priors = _coarse_priors(
                et_frame, eye_frame, et_audit, eye_audit, settings["coarse_sample_hz"])
            base["n_coarse_priors"] = len(priors)
            if not priors:
                raise ValueError("no coarse pupil-content or endpoint prior could be formed")

            combined = _best_fit(et_frame, eye_frame, PUPIL_GROUPS, priors, settings)
            left = _best_fit(et_frame, eye_frame, {"left-pupil": ["left-pupil"]},
                             priors, settings)
            right = _best_fit(et_frame, eye_frame, {"right-pupil": ["right-pupil"]},
                              priors, settings)
            combined_status, combined_reasons = _fit_status(combined, settings)
            left_status, left_reasons = _fit_status(left, settings)
            right_status, right_reasons = _fit_status(right, settings)
            base.update(_fit_summary("left", left, left_status))
            base.update(_fit_summary("right", right, right_status))
            windows.extend(_window_rows(job, "combined", combined, settings))
            windows.extend(_window_rows(job, "left", left, settings))
            windows.extend(_window_rows(job, "right", right, settings))

            reference = float((et_audit["start_s"] + et_audit["end_s"]) / 2.0)
            base["reference_et_s"] = reference
            reliable = {"good", "review", "offset_only"}
            eye_conflict = False
            exactly_one_good = ((left_status == "good")
                                != (right_status == "good"))
            preferred_eye_source = (
                ("left_eye" if left_status == "good" else "right_eye")
                if exactly_one_good else None)
            selection_warning = ""
            if left_status in reliable and right_status in reliable:
                offset_difference = abs(_mapping_at(left, reference) -
                                        _mapping_at(right, reference))
                base["eye_offset_difference_s"] = offset_difference
                if left_status != "offset_only" and right_status != "offset_only":
                    drift_difference = abs(float(left["drift_ppm"]) -
                                           float(right["drift_ppm"]))
                    base["eye_drift_difference_ppm"] = drift_difference
                else:
                    drift_difference = 0.0
                eyes_disagree = bool(
                    offset_difference > settings["eye_offset_difference_s"]
                    or drift_difference > settings["eye_drift_difference_ppm"])
                if eyes_disagree:
                    # Facial/camera asymmetry commonly makes one eye much
                    # weaker. A single unequivocally good eye outranks a
                    # review/offset-only eye; only equally ranked evidence is
                    # allowed to block automatic selection.
                    if exactly_one_good:
                        weaker = "right" if preferred_eye_source == "left_eye" else "left"
                        selection_warning = (
                            f"selected the good {preferred_eye_source.replace('_', ' ')}; "
                            f"the weaker {weaker} eye supported a different mapping")
                        add_issue(job, "warning", "weaker_eye_mapping_disagrees",
                                  selection_warning)
                    else:
                        eye_conflict = True
                        add_issue(job, "error", "left_right_mapping_conflict",
                                  f"left/right mappings differ by {offset_difference:.3f} s "
                                  f"and {drift_difference:.1f} ppm")

            candidates = [(combined, combined_status, "both_eyes"),
                          (left, left_status, "left_eye"),
                          (right, right_status, "right_eye")]
            candidates = [item for item in candidates if item[1] in reliable]
            if preferred_eye_source is not None:
                candidates = [item for item in candidates
                              if item[2] == preferred_eye_source]
            selected, selected_status, selected_source = (
                max(candidates, key=lambda item: _fit_rank(item[0]))
                if candidates else (None, "unavailable", "none"))

            if eye_conflict:
                base.update({"mapping_status": "eye_conflict", "mapping_source": "none",
                             "mapping_reason": "reliable left/right mappings disagree"})
            elif selected is None:
                invalid_reasons = []
                for status, reasons in ((combined_status, combined_reasons),
                                        (left_status, left_reasons),
                                        (right_status, right_reasons)):
                    if status == "invalid":
                        invalid_reasons.extend(reasons)
                if invalid_reasons:
                    base["mapping_status"] = "invalid_affine"
                    base["mapping_reason"] = "; ".join(dict.fromkeys(invalid_reasons))
                    add_issue(job, "error", "invalid_content_mapping",
                              base["mapping_reason"])
                else:
                    base["mapping_reason"] = "; ".join(
                        combined_reasons or left_reasons or right_reasons)
            else:
                offset_only = selected_status == "offset_only"
                scale = 1.0 if offset_only else float(selected["scale"])
                intercept = (float(np.median([
                    window["offset_s"] for window in selected.get("windows", [])
                ])) if offset_only else float(selected["intercept_s"]))
                status_name = ("offset_only" if offset_only else
                               ("content_review" if selected_status == "review"
                                else "content_affine"))
                grade = ("C" if selected_status in {"review", "offset_only"}
                         else ("A" if selected_source == "both_eyes"
                               and left_status == "good" and right_status == "good"
                               else "B"))
                reasons = ({"both_eyes": combined_reasons, "left_eye": left_reasons,
                            "right_eye": right_reasons}[selected_source])
                base.update({
                    "mapping_status": status_name, "mapping_source": selected_source,
                    "usable": True, "quality_grade": grade,
                    "review_required": selected_status != "good",
                    "mapping_reason": ("local pupil-content windows" if not reasons
                                       else "; ".join(reasons)),
                    "intercept_s": intercept, "scale": scale,
                    "drift_ppm": np.nan if offset_only else (scale - 1.0) * 1e6,
                    "offset_at_reference_s": intercept + (scale - 1.0) * reference,
                    "coarse_prior_s": selected.get("prior_offset_s"),
                    "coarse_prior_source": selected.get("coarse_prior_source"),
                    "n_windows": selected.get("n_windows", 0),
                    "n_good_windows": selected.get("n_good_windows", 0),
                    "fit_inliers": selected.get("fit_inliers", 0),
                    "n_eyes": selected.get("n_groups", 0),
                    "eyes": "|".join(selected.get("groups", [])),
                    "median_peak_correlation": selected.get("median_peak_correlation"),
                    "median_peak_margin": selected.get("median_peak_margin"),
                    "median_eye_spread_s": selected.get("median_group_spread_s"),
                    "fit_residual_mad_s": selected.get("fit_residual_mad_s"),
                    "fit_residual_p95_s": selected.get("fit_residual_p95_s"),
                    "good_window_ratio": selected.get("good_window_ratio", 0.0),
                    "temporal_coverage_ratio": selected.get("temporal_coverage_ratio", 0.0),
                })
                if selection_warning:
                    base["mapping_reason"] = (
                        f"{base['mapping_reason']}; {selection_warning}")
                amplitude_results = {}
                for side in ("left", "right"):
                    amplitude = _amplitude_regression(
                        et_frame, eye_frame, side, intercept, scale, settings["max_gap_s"])
                    amplitude_results[side] = amplitude
                    base.update({
                        f"{side}_amplitude_samples": amplitude["samples"],
                        f"{side}_sqrt_area_intercept": amplitude["intercept"],
                        f"{side}_sqrt_area_per_mm": amplitude["slope"],
                        f"{side}_amplitude_r": amplitude["r"],
                        f"{side}_amplitude_r2": amplitude["r2"],
                    })
                    if np.isfinite(amplitude["slope"]) and amplitude["slope"] <= 0:
                        add_issue(job, "warning", "nonpositive_amplitude_slope",
                                  f"{side} sqrt-area/diameter slope is not positive")
                selected_sides = ({"left_eye": ("left",), "right_eye": ("right",)}
                                  .get(selected_source, ("left", "right")))
                amplitude_rs = [amplitude_results[side]["r"] for side in selected_sides
                                if np.isfinite(amplitude_results[side]["r"])]
                if not amplitude_rs or max(amplitude_rs) < settings["min_amplitude_r"]:
                    amplitude_reason = (
                        "aligned pupil amplitudes have weak global linear association")
                    add_issue(job, "warning", "weak_amplitude_relation", amplitude_reason)
                    base["review_required"] = True
                    base["quality_grade"] = "C"
                    if base["mapping_status"] == "content_affine":
                        base["mapping_status"] = "content_review"
                    base["mapping_reason"] = f"{base['mapping_reason']}; {amplitude_reason}"

            warnings = list(dict.fromkeys(
                combined_reasons + left_reasons + right_reasons))
            if selection_warning:
                warnings.append(selection_warning)
            base["warnings"] = " | ".join(warnings)
            for reason in warnings:
                add_issue(job, "warning", "fit_qc", reason)
            mappings.append(base)
        except Exception as error:  # keep one visible row per discovered job
            base.update({
                "processing_status": "error", "mapping_status": "error",
                "mapping_reason": f"{type(error).__name__}: {error}",
            })
            add_issue(job, "error", "processing_error", base["mapping_reason"])
            mappings.append(base)

    return (
        pd.DataFrame(mappings, columns=ALIGNMENT_COLUMNS),
        pd.DataFrame(windows, columns=WINDOW_COLUMNS),
        pd.DataFrame(issues, columns=ISSUE_COLUMNS),
    )


def save_et_eyeball_alignment_results(alignments, windows, issues, *,
                                       alignment_csv, windows_csv, issues_csv):
    """Write the alignment, local-window and issue tables."""
    paths = tuple(Path(path).resolve() for path in (
        alignment_csv, windows_csv, issues_csv
    ))
    for frame, path in zip((alignments, windows, issues), paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return paths


