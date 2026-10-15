"""Estimate clock mappings between native ET recordings and their XDF
streams.

Dependencies: native Tobii recordings, matching XDF files, NumPy and
pandas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from helpers.lsl_et_sync import et_loaders, lsl_loaders
from helpers.lsl_et_sync import offset as offset_core


GROUP_COLUMNS = ["project", "pid", "visit"]
DISCOVERY_COLUMNS = [
    'project', 'pid', 'visit', 'et_recording_path',
    'xdf_path', 'discovery_status',
]
IMU_GROUPS = dict(offset_core.IMU_GROUPS)
OCULAR_GROUPS = dict(offset_core.ET_GROUPS)
ALL_GROUPS = {**IMU_GROUPS, **OCULAR_GROUPS}

DEFAULT_WINDOW_S = 60.0
DEFAULT_STEP_S = 30.0
DEFAULT_LOCAL_RESIDUAL_SEARCH_RADIUS_S = 10.0
DEFAULT_MAX_GAP_S = 0.2
DEFAULT_MIN_CORRELATION = 0.30
DEFAULT_MIN_PEAK_MARGIN = 0.02
DEFAULT_MIN_GOOD_WINDOWS = 3
DEFAULT_REVIEW_DRIFT_PPM = 2_000.0
DEFAULT_MAX_DRIFT_PPM = 10_000.0
DEFAULT_MAX_RESIDUAL_P95_S = 0.20
DEFAULT_MAX_CLOCK_JUMP_S = 0.50
DEFAULT_MODALITY_OFFSET_DIFFERENCE_S = 0.20
DEFAULT_MODALITY_DRIFT_DIFFERENCE_PPM = 1_000.0

ALIGNMENT_COLUMNS = [
    'project', 'pid', 'visit', 'mapping_status',
    'usable', 'quality_grade', 'review_required', 'mapping_reason',
    'intercept_s', 'scale', 'drift_ppm', 'n_good_windows',
    'fit_residual_p95_s', 'et_start_s', 'et_end_s', 'warnings',
    'lsl_start_s', 'lsl_end_s',
]


def _normalise_filter(values):
    if values is None:
        return None
    return {str(value) for value in values}


def _pid_visit(name: str):
    parts = str(name).split("_")
    return (parts[0], parts[1]) if len(parts) >= 2 else None


def discover_et_lsl_jobs(data_root, project, participants=None, visits=None):
    root = Path(data_root)
    participant_filter = _normalise_filter(participants)
    visit_filter = _normalise_filter(visits)
    et_by_key, xdf_by_key = {}, {}

    if root.is_dir():
        for participant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            pid = participant_dir.name
            if participant_filter is not None and pid not in participant_filter:
                continue
            et_dir = participant_dir / "ET"
            if et_dir.is_dir():
                for recording in sorted(path for path in et_dir.iterdir() if path.is_dir()):
                    if not ((recording / et_loaders.GAZE_FILE).is_file()
                            or (recording / et_loaders.IMU_FILE).is_file()):
                        continue
                    key = _pid_visit(recording.name)
                    if key and key[0] == pid:
                        et_by_key.setdefault(key, []).append(recording.resolve())
            lsl_dir = participant_dir / "LSL"
            if lsl_dir.is_dir():
                for xdf in sorted(lsl_dir.glob("*.xdf")):
                    key = _pid_visit(xdf.name)
                    if key and key[0] == pid:
                        xdf_by_key.setdefault(key, []).append(xdf.resolve())

    rows = []
    for pid, visit in sorted(set(et_by_key) | set(xdf_by_key)):
        if visit_filter is not None and visit not in visit_filter:
            continue
        et_candidates = sorted(set(et_by_key.get((pid, visit), [])))
        xdf_candidates = sorted(set(xdf_by_key.get((pid, visit), [])))
        if len(et_candidates) == 1 and len(xdf_candidates) == 1:
            status = "ready"
        elif not et_candidates:
            status = "missing_et_raw"
        elif not xdf_candidates:
            status = "missing_lsl"
        elif len(et_candidates) > 1:
            status = "ambiguous_et_raw"
        else:
            status = "ambiguous_lsl"
        rows.append({
            "project": str(project), "pid": pid, "visit": visit,
            "et_recording_path": str(et_candidates[0]) if len(et_candidates) == 1 else "",
            "xdf_path": str(xdf_candidates[0]) if len(xdf_candidates) == 1 else "",
            "discovery_status": status,
        })
    return pd.DataFrame(rows, columns=DISCOVERY_COLUMNS)


def _timestamp_values(frame, *, et_side):
    if frame is None or frame.empty:
        return np.array([], dtype=float)
    if et_side:
        return pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(float)
    if isinstance(frame.index, pd.DatetimeIndex):
        values = frame.index.view("int64").astype(float) / 1e9
        values[np.asarray(frame.index.isna())] = np.nan
        return values
    return pd.to_numeric(pd.Index(frame.index), errors="coerce").to_numpy(float)


def _timestamp_audit(frame, *, et_side):
    values = _timestamp_values(frame, et_side=et_side)
    finite = values[np.isfinite(values)]
    return {
        "start_s": float(finite[0]) if len(finite) else np.nan,
        "end_s": float(finite[-1]) if len(finite) else np.nan,
        "nonincreasing_timestamps": int(np.sum(np.diff(finite) <= 0)),
    }


def _et_timestamp_audit(gaze, imu):
    audits = []
    for frame, groups in ((gaze, OCULAR_GROUPS), (imu, IMU_GROUPS)):
        if frame is None or frame.empty:
            continue
        for features in groups.values():
            if not all(feature in frame.columns for feature in features):
                continue
            values = frame[list(features)].apply(pd.to_numeric, errors="coerce")
            mask = np.isfinite(values.to_numpy(float)).all(axis=1)
            if mask.any():
                audits.append(_timestamp_audit(frame.loc[mask], et_side=True))
    if not audits:
        return _timestamp_audit(None, et_side=True)
    return {
        "start_s": float(min(audit["start_s"] for audit in audits)),
        "end_s": float(max(audit["end_s"] for audit in audits)),
        "nonincreasing_timestamps": int(max(
            audit["nonincreasing_timestamps"] for audit in audits)),
    }


def _group_times(frame, features, *, et_side):
    if frame is None or frame.empty or not all(name in frame for name in features):
        return np.array([], dtype=float)
    numeric = frame[list(features)].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(numeric.to_numpy(float)).all(axis=1)
    return _timestamp_values(frame.loc[mask], et_side=et_side)


def estimate_group_sample_rates(et_frame, lsl_frame):
    rates = {}
    for group, features in ALL_GROUPS.items():
        side_rates = []
        for frame, et_side in ((et_frame, True), (lsl_frame, False)):
            times = _group_times(frame, features, et_side=et_side)
            diffs = np.diff(np.unique(times[np.isfinite(times)]))
            diffs = diffs[diffs > 0]
            if len(diffs):
                side_rates.append(float(1.0 / np.median(diffs)))
        if len(side_rates) == 2:
            cap = float(offset_core.GROUP_SAMPLE_HZ.get(group, 50.0))
            rates[group] = max(1.0, min(cap, *side_rates))
    return rates


def _combine_et(gaze, imu):
    available = [frame for frame in (gaze, imu) if frame is not None and not frame.empty]
    if not available:
        return None
    combined = pd.concat(available, ignore_index=True, sort=False)
    combined["timestamp"] = pd.to_numeric(combined["timestamp"], errors="coerce")
    return combined.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _subset(frame, groups, *, et_side):
    features = sorted({column for columns in groups.values() for column in columns})
    keep = [column for column in features if column in frame.columns]
    if et_side:
        keep = ["timestamp"] + keep
    return frame.loc[:, keep].copy()


def _fit_status(fit, *, min_good_windows, review_drift_ppm,
                max_drift_ppm, max_residual_p95_s, max_clock_jump_s):
    if not fit or not all(key in fit for key in ("intercept_s", "scale")):
        return "unavailable", ["no fitted mapping"]
    reasons = []
    good = int(fit.get("n_good_windows", 0))
    inliers = int(fit.get("fit_inliers", 0))
    if good < min_good_windows or inliers < min_good_windows:
        reasons.append(f"only {good} good windows / {inliers} fit inliers")
    drift = float(fit.get("raw_drift_ppm", fit.get("drift_ppm", np.nan)))
    if not np.isfinite(drift) or abs(drift) > max_drift_ppm:
        reasons.append(f"drift {drift:.1f} ppm exceeds hard limit")
    residual = float(fit.get("fit_residual_p95_s", np.inf))
    if not np.isfinite(residual) or residual > max_residual_p95_s:
        reasons.append(f"residual p95 {residual:.3f} s exceeds limit")
    jump = _max_clock_jump(fit)
    if np.isfinite(jump) and jump > max_clock_jump_s:
        reasons.append(f"possible clock jump {jump:.3f} s")
    if reasons:
        return "invalid", reasons
    review = []
    if abs(drift) > review_drift_ppm:
        review.append(f"large drift {drift:.1f} ppm")
    if int(fit.get("n_groups", 0)) < 2:
        review.append("mapping supported by one sensor group")
    good_ratio = float(fit.get("good_window_ratio", 0.0))
    temporal_ratio = float(fit.get("temporal_coverage_ratio", 0.0))
    if good_ratio < 0.25:
        review.append(f"only {good_ratio:.1%} of windows passed QC")
    if temporal_ratio < 0.50:
        review.append(f"good windows cover only {temporal_ratio:.1%} of the overlap")
    return ("review" if review else "good"), review


def _max_clock_jump(fit):
    windows = sorted(fit.get("windows", []), key=lambda row: row.get("center_et_s", 0))
    if len(windows) < 2 or "intercept_s" not in fit:
        return np.nan
    x = np.asarray([row["center_et_s"] for row in windows], dtype=float)
    y = np.asarray([row["offset_s"] for row in windows], dtype=float)
    trend = float(fit["intercept_s"]) + (float(fit["scale"]) - 1.0) * x
    residual = y - trend
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center)))
    keep = np.abs(residual - center) <= max(0.20, 4.0 * 1.4826 * mad)
    residual = residual[keep]
    return float(np.max(np.abs(np.diff(residual)))) if len(residual) >= 2 else np.nan


def _mapping_at(fit, et_time):
    return float(fit["intercept_s"] + (fit["scale"] - 1.0) * et_time)


def _legacy_row(legacy, pid, visit):
    if (legacy is None or legacy.empty
            or not {"pid", "visit"}.issubset(legacy.columns)):
        return None
    mask = legacy["pid"].astype(str).eq(str(pid)) & legacy["visit"].astype(str).eq(str(visit))
    matches = legacy.loc[mask]
    return matches.iloc[-1] if len(matches) else None


def _finite(value, default=np.nan):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _legacy_mapping(row):
    if row is None:
        return None
    intercept = _finite(row.get("offset_LSL_ET"))
    scale = _finite(row.get("sync_scale"), 1.0)
    grade = str(row.get("sync_quality_grade", "")).upper()
    complete = str(row.get("sync_processing_status", "complete")).lower() == "complete"
    if not np.isfinite(intercept) or not np.isfinite(scale) or scale <= 0:
        return None
    return {
        "intercept_s": intercept, "scale": scale,
        "drift_ppm": (scale - 1.0) * 1e6,
        "reliable": bool(complete and grade in {"A", "B"}),
    }


def _run_fit(et_frame, lsl_frame, groups, prior, rates, settings):
    et_sub = _subset(et_frame, groups, et_side=True)
    lsl_sub = _subset(lsl_frame, groups, et_side=False)
    if len(et_sub.columns) <= 1 or not len(lsl_sub.columns):
        return None
    return offset_core.estimate_windowed_alignment(
        et_sub, lsl_sub, initial_offset_s=prior,
        window_s=settings["window_s"], step_s=settings["step_s"],
        search_radius_s=settings["local_residual_search_radius_s"],
        max_gap_s=settings["max_gap_s"],
        min_peak_correlation=settings["min_peak_correlation"],
        min_peak_margin=settings["min_peak_margin"],
        sample_hz_by_group=rates, max_abs_drift_ppm=None, verbose=False,
        min_groups_per_window=2,
    )


def build_et_lsl_alignments(
    jobs,
    *,
    legacy_results=None,
    window_s=DEFAULT_WINDOW_S,
    step_s=DEFAULT_STEP_S,
    local_residual_search_radius_s=DEFAULT_LOCAL_RESIDUAL_SEARCH_RADIUS_S,
    max_gap_s=DEFAULT_MAX_GAP_S,
    min_peak_correlation=DEFAULT_MIN_CORRELATION,
    min_peak_margin=DEFAULT_MIN_PEAK_MARGIN,
    min_good_windows=DEFAULT_MIN_GOOD_WINDOWS,
    review_drift_ppm=DEFAULT_REVIEW_DRIFT_PPM,
    max_drift_ppm=DEFAULT_MAX_DRIFT_PPM,
    max_residual_p95_s=DEFAULT_MAX_RESIDUAL_P95_S,
    max_clock_jump_s=DEFAULT_MAX_CLOCK_JUMP_S,
    modality_offset_difference_s=DEFAULT_MODALITY_OFFSET_DIFFERENCE_S,
    modality_drift_difference_ppm=DEFAULT_MODALITY_DRIFT_DIFFERENCE_PPM,
):
    settings = dict(
        window_s=float(window_s), step_s=float(step_s),
        local_residual_search_radius_s=float(local_residual_search_radius_s),
        max_gap_s=float(max_gap_s),
        min_peak_correlation=float(min_peak_correlation),
        min_peak_margin=float(min_peak_margin),
    )
    if any(settings[name] <= 0 for name in
           ("window_s", "step_s", "local_residual_search_radius_s", "max_gap_s")):
        raise ValueError("window, step, search radius and max gap must be positive")
    if int(min_good_windows) < 1:
        raise ValueError("min_good_windows must be at least 1")
    if not (0 <= float(review_drift_ppm) <= float(max_drift_ppm)):
        raise ValueError("drift thresholds must satisfy 0 <= review <= max")
    mappings = []
    for _, job_series in jobs.iterrows():
        job = job_series.to_dict()
        base = {column: np.nan for column in ALIGNMENT_COLUMNS}
        base.update({
            "project": job.get("project", ""), "pid": job.get("pid", ""),
            "visit": job.get("visit", ""), "mapping_status": "unavailable",
            "usable": False, "quality_grade": "D", "review_required": True,
            "warnings": "",
        })
        if job.get("discovery_status") != "ready":
            base["mapping_reason"] = str(job.get("discovery_status", "not ready"))
            mappings.append(base)
            continue
        try:
            recording_path = Path(job["et_recording_path"])
            gaze = et_loaders.load_et_recording(recording_path)
            imu = et_loaders.load_imu_recording(recording_path)
            et_frame = _combine_et(gaze, imu)
            lsl_frame, _ = lsl_loaders.load_lsl_xdf(Path(job["xdf_path"]))
            if et_frame is None or et_frame.empty:
                raise ValueError("ET raw gaze and IMU streams are both unavailable")
            if lsl_frame is None or lsl_frame.empty:
                raise ValueError("ET stream is unavailable in the XDF")
            et_audit = _et_timestamp_audit(gaze, imu)
            lsl_audit = _timestamp_audit(lsl_frame, et_side=False)
            for prefix, audit in (("et", et_audit), ("lsl", lsl_audit)):
                for key in ("start_s", "end_s"):
                    base[f"{prefix}_{key}"] = audit[key]
            rates = estimate_group_sample_rates(et_frame, lsl_frame)
            legacy = _legacy_mapping(_legacy_row(legacy_results, job["pid"], job["visit"]))
            combined = _run_fit(et_frame, lsl_frame, ALL_GROUPS, None, rates, settings)
            if combined and combined.get("n_good_windows", 0):
                prior = float(combined["prior_offset_s"])
            elif legacy and legacy["reliable"]:
                et_middle = (et_audit["start_s"] + et_audit["end_s"]) / 2.0
                prior = _mapping_at(legacy, et_middle)
                combined = _run_fit(et_frame, lsl_frame, ALL_GROUPS, prior, rates, settings)
            else:
                prior = np.nan
            imu_fit = _run_fit(
                et_frame, lsl_frame, IMU_GROUPS,
                prior if np.isfinite(prior) else None, rates, settings)
            ocular_fit = _run_fit(
                et_frame, lsl_frame, OCULAR_GROUPS,
                prior if np.isfinite(prior) else None, rates, settings)
            classifiers = dict(
                min_good_windows=int(min_good_windows),
                review_drift_ppm=float(review_drift_ppm),
                max_drift_ppm=float(max_drift_ppm),
                max_residual_p95_s=float(max_residual_p95_s),
                max_clock_jump_s=float(max_clock_jump_s),
            )
            combined_status, combined_reasons = _fit_status(combined, **classifiers)
            imu_status, imu_reasons = _fit_status(imu_fit, **classifiers)
            ocular_status, ocular_reasons = _fit_status(ocular_fit, **classifiers)
            reference = float(np.nanmedian([
                value for value in (et_audit["start_s"], et_audit["end_s"])
                if np.isfinite(value)]))
            modality_conflict = False
            if imu_status in {"good", "review"} and ocular_status in {"good", "review"}:
                offset_difference = abs(_mapping_at(imu_fit, reference) -
                                        _mapping_at(ocular_fit, reference))
                drift_difference = abs(float(imu_fit["drift_ppm"]) -
                                       float(ocular_fit["drift_ppm"]))
                modality_conflict = bool(
                    offset_difference > modality_offset_difference_s
                    or drift_difference > modality_drift_difference_ppm)
            selected, selected_source, selected_status = None, "none", "unavailable"
            if not modality_conflict and combined_status in {"good", "review"}:
                selected, selected_source, selected_status = combined, "combined_content", combined_status
            elif not modality_conflict and imu_status in {"good", "review"}:
                selected, selected_source, selected_status = imu_fit, "imu_content", imu_status
            elif not modality_conflict and ocular_status in {"good", "review"}:
                selected, selected_source, selected_status = ocular_fit, "ocular_content", ocular_status
            elif modality_conflict:
                base["mapping_status"] = "conflict_review"
                base["mapping_reason"] = "reliable IMU and ocular mappings disagree"
            elif legacy and legacy["reliable"]:
                selected, selected_source, selected_status = legacy, "legacy_fallback", "review"
            hard_timestamp_error = bool(
                et_audit["nonincreasing_timestamps"] or lsl_audit["nonincreasing_timestamps"])
            if selected is not None and not hard_timestamp_error:
                if selected_status == "good":
                    mapping_status = "content_affine"
                elif selected_source != "legacy_fallback":
                    mapping_status = "content_review"
                else:
                    mapping_status = "legacy_fallback"

                if (selected_source == "combined_content"
                        and selected_status == "good"
                        and imu_status == "good"
                        and ocular_status == "good"):
                    quality_grade = "A"
                elif selected_status == "good":
                    quality_grade = "B"
                else:
                    quality_grade = "C"

                base.update({
                    "mapping_status": mapping_status,
                    "usable": True,
                    "quality_grade": quality_grade,
                    "review_required": bool(selected_status != "good"),
                    "intercept_s": selected["intercept_s"], "scale": selected["scale"],
                    "drift_ppm": selected.get("drift_ppm", (selected["scale"] - 1) * 1e6),
                    "mapping_reason": ("local content windows" if selected_source != "legacy_fallback"
                                       else "content mapping unavailable; reliable legacy fallback"),
                })
                if selected_source != "legacy_fallback":
                    base["n_good_windows"] = selected.get("n_good_windows", 0)
                    base["fit_residual_p95_s"] = selected.get("fit_residual_p95_s")
            elif hard_timestamp_error:
                base.update({"mapping_status": "invalid_timestamps",
                             "mapping_reason": "non-increasing source timestamps"})
            elif base["mapping_status"] != "conflict_review":
                base["mapping_reason"] = "; ".join(combined_reasons or
                                                    imu_reasons or ocular_reasons)
            base["warnings"] = " | ".join(dict.fromkeys(
                combined_reasons + imu_reasons + ocular_reasons))
            mappings.append(base)
        except Exception as error:
            base.update({
                "mapping_status": "error",
                "mapping_reason": f"{type(error).__name__}: {error}",
            })
            mappings.append(base)
    return pd.DataFrame(mappings, columns=ALIGNMENT_COLUMNS)


def save_et_lsl_alignment_results(alignments, *, alignment_csv):
    path = Path(alignment_csv).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    alignments.to_csv(path, index=False)
    return path
