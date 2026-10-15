"""Estimate clock mappings between native ET recordings and Eyeball pupil
traces.

Dependencies: native Tobii recordings, pupil CSVs from Eyeball 00, NumPy
and pandas."""

from __future__ import annotations

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
    'project', 'pid', 'visit', 'et_recording_path',
    'pupil_csv_path', 'discovery_status',
]

ALIGNMENT_COLUMNS = [
    'project', 'pid', 'visit', 'mapping_status',
    'usable', 'quality_grade', 'review_required', 'mapping_reason',
    'intercept_s', 'scale', 'drift_ppm', 'n_good_windows',
    'fit_residual_p95_s', 'et_start_s', 'et_end_s', 'mapping_source',
    'pupil_csv_path',
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
            "discovery_status": status,
        })
    return pd.DataFrame(rows, columns=DISCOVERY_COLUMNS)


def _timestamp_audit(values):
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    finite = numeric[np.isfinite(numeric)]
    return {
        "start_s": float(finite[0]) if len(finite) else np.nan,
        "end_s": float(finite[-1]) if len(finite) else np.nan,
        "nonincreasing_timestamps": int(np.sum(np.diff(finite) <= 0)),
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

    et_prepared = et_prepared[np.isfinite(et_prepared["timestamp"])].copy()
    eye_prepared = eye_prepared[~eye_prepared.index.isna()].copy()
    return et_prepared, eye_prepared, et_audit, eye_audit


def _deduplicate_priors(priors, tolerance_s=0.25, limit=8):
    output = []
    for value in priors:
        if not np.isfinite(value):
            continue
        if any(abs(float(value) - previous) <= tolerance_s for previous in output):
            continue
        output.append(float(value))
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
    priors.extend(candidate[0] for candidate in candidates[:5])
    priors.extend([
        eye_audit["end_s"] - et_audit["end_s"],
        eye_audit["start_s"] - et_audit["start_s"],
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
    for prior in priors:
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
    times, values = unique, values[keep]
    right = np.searchsorted(times, grid, side="left")
    inside = (right > 0) & (right < len(times))
    left_index = np.clip(right - 1, 0, len(times) - 1)
    right_index = np.clip(right, 0, len(times) - 1)
    allowed = inside & ((times[right_index] - times[left_index]) <= max_gap_s)
    output = np.interp(grid, times, values)
    output[~allowed] = np.nan
    return output


def _amplitude_correlation(et_frame, eye_frame, side, intercept, scale,
                           max_gap_s, sample_hz=10.0):
    column = f"{side}-pupil"
    et_times = et_frame["timestamp"].to_numpy(float) * scale + intercept
    et_values = et_frame[column].to_numpy(float)
    eye_times = eye_frame.index.view("int64").astype(float) / 1e9
    eye_values = eye_frame[column].to_numpy(float)
    start = max(np.nanmin(et_times), np.nanmin(eye_times))
    end = min(np.nanmax(et_times), np.nanmax(eye_times))
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return np.nan
    grid = np.arange(start, end, 1.0 / float(sample_hz))
    diameter = _interpolate_with_gap_mask(et_times, et_values, grid, max_gap_s)
    sqrt_area = _interpolate_with_gap_mask(eye_times, eye_values, grid, max_gap_s)
    valid = np.isfinite(diameter) & np.isfinite(sqrt_area)
    if valid.sum() < 100:
        return np.nan
    x, y = diameter[valid], sqrt_area[valid]
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan


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

    mappings = []

    for _, job_series in jobs.iterrows():
        job = job_series.to_dict()
        base = {column: np.nan for column in ALIGNMENT_COLUMNS}
        base.update({
            "project": job.get("project", ""), "pid": job.get("pid", ""),
            "visit": job.get("visit", ""),
            "mapping_status": "unavailable", "mapping_source": "none",
            "usable": False, "quality_grade": "D", "review_required": True,
        })
        if job.get("discovery_status") != "ready":
            reason = str(job.get("discovery_status", "not ready"))
            base["mapping_reason"] = reason
            mappings.append(base)
            continue
        try:
            recording_path = Path(job["et_recording_path"])
            pupil_path = Path(job["pupil_csv_path"])
            et_frame, eye_frame, et_audit, eye_audit = _load_sources(
                recording_path, pupil_path)
            for name in ("start_s", "end_s"):
                base[f"et_{name}"] = et_audit[name]

            base.update({
                "pupil_csv_path": str(pupil_path),
            })
            hard_timestamp_error = bool(
                et_audit["nonincreasing_timestamps"]
                or eye_audit["nonincreasing_timestamps"])
            if hard_timestamp_error:
                base.update({"mapping_status": "invalid_timestamps",
                             "mapping_reason": "source timestamps are not strictly increasing"})
                mappings.append(base)
                continue

            priors = _coarse_priors(
                et_frame, eye_frame, et_audit, eye_audit, settings["coarse_sample_hz"])
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

            reference = float((et_audit["start_s"] + et_audit["end_s"]) / 2.0)
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
                if left_status != "offset_only" and right_status != "offset_only":
                    drift_difference = abs(float(left["drift_ppm"]) -
                                           float(right["drift_ppm"]))
                else:
                    drift_difference = 0.0
                eyes_disagree = bool(
                    offset_difference > settings["eye_offset_difference_s"]
                    or drift_difference > settings["eye_drift_difference_ppm"])
                if eyes_disagree:
                    if exactly_one_good:
                        weaker = "right" if preferred_eye_source == "left_eye" else "left"
                        selection_warning = (
                            f"selected the good {preferred_eye_source.replace('_', ' ')}; "
                            f"the weaker {weaker} eye supported a different mapping")
                    else:
                        eye_conflict = True

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
                    "n_good_windows": selected.get("n_good_windows", 0),
                    "fit_residual_p95_s": selected.get("fit_residual_p95_s"),
                })
                if selection_warning:
                    base["mapping_reason"] = (
                        f"{base['mapping_reason']}; {selection_warning}")
                selected_sides = ({"left_eye": ("left",), "right_eye": ("right",)}
                                  .get(selected_source, ("left", "right")))
                amplitude_rs = [
                    _amplitude_correlation(
                        et_frame, eye_frame, side, intercept, scale, settings["max_gap_s"])
                    for side in selected_sides
                ]
                amplitude_rs = [value for value in amplitude_rs if np.isfinite(value)]
                if not amplitude_rs or max(amplitude_rs) < settings["min_amplitude_r"]:
                    amplitude_reason = (
                        "aligned pupil amplitudes have weak global linear association")
                    base["review_required"] = True
                    base["quality_grade"] = "C"
                    if base["mapping_status"] == "content_affine":
                        base["mapping_status"] = "content_review"
                    base["mapping_reason"] = f"{base['mapping_reason']}; {amplitude_reason}"

            mappings.append(base)
        except Exception as error:
            base.update({
                "mapping_status": "error",
                "mapping_reason": f"{type(error).__name__}: {error}",
            })
            mappings.append(base)

    return pd.DataFrame(mappings, columns=ALIGNMENT_COLUMNS)


def save_et_eyeball_alignment_results(alignments, *, alignment_csv):
    path = Path(alignment_csv).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    alignments.to_csv(path, index=False)
    return path

