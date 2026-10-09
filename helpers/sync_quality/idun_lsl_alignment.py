"""Align native IDUN raw EEG to cap EEG on the LSL time base.

The preceding raw/LSL repair verifies the IDUN samples and records a coarse
arrival-time offset.  This module uses content shared with the simultaneously
recorded cap EEG to estimate the affine mapping::

    t_cap_lsl = intercept_s + scale * t_idun_raw_relative

The cloud-arrival lower envelope from the repair stage is used only to center
the lag search.  It is never accepted as the final offset.  Matching uses a
robust short-time activity envelope, with T7-T8 as the primary cap signal and
near-ear alternatives as independent support.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .publication_contract import required_publication_columns


PRIMARY_CAP_FEATURE = "T7-T8"
CAP_FEATURES = {
    "T7-T8": ("T7", "T8"),
    "T8-M1": ("T8", "M1"),
    "T7-M2": ("T7", "M2"),
    "M1-M2": ("M1", "M2"),
    "T7": ("T7",),
    "T8": ("T8",),
}


MAPPING_COLUMNS = [
    "project", "pid", "visit",
    "mapping_status", "mapping_source", "usable", "quality_grade",
    "review_required", "selected_cap_feature", "intercept_s", "scale",
    "drift_ppm", "idun_raw_time_origin_unix_s", "idun_raw_duration_s",
    "fit_support_start_idun_raw_s", "fit_support_end_idun_raw_s",
    "fit_support_start_lsl_s", "fit_support_end_lsl_s", "cap_lsl_start_s",
    "cap_lsl_end_s", "overlap_duration_s", "coarse_prior_offset_s",
    "coarse_prior_source", "coarse_lag_s", "n_windows", "n_good_windows",
    "n_fit_inlier_windows", "median_peak_correlation",
    "median_peak_margin", "fit_residual_median_s", "fit_residual_p95_s",
    "max_local_clock_jump_s", "support_channel_count",
    "max_channel_offset_difference_s", "max_channel_drift_difference_ppm",
    "channel_agreement", "raw_csv_path", "xdf_path",
    "mapping_reason", "warnings",
]


CHANNEL_COLUMNS = [
    "project", "pid", "visit", "cap_feature", "fit_status",
    "intercept_s", "scale", "drift_ppm", "coarse_lag_s", "n_windows",
    "n_good_windows", "n_fit_inlier_windows", "median_peak_correlation",
    "median_peak_margin", "fit_residual_median_s", "fit_residual_p95_s",
    "max_local_clock_jump_s", "selection_score", "selected",
]


WINDOW_COLUMNS = [
    "project", "pid", "visit", "cap_feature", "window_index",
    "idun_raw_center_s", "prior_lsl_center_s", "lag_s", "peak_correlation",
    "runner_up_correlation", "peak_margin", "accepted", "rejection_reason",
    "fit_inlier", "fit_residual_s",
]


ISSUE_COLUMNS = [
    "project", "pid", "visit", "severity", "issue", "detail",
]


PUBLICATION_DATA_COLUMNS = list(required_publication_columns("IDUN"))


SYNCHRONIZED_SUMMARY_COLUMNS = [
    "project", "pid", "visit",
    "export_status", "idun_mapping_status", "idun_quality_grade",
    "synchronized_csv_path", "source_raw_csv_path", "sample_count",
    "lsl_timestamp_count", "lsl_supported_sample_count",
    "lsl_extrapolated_sample_count", "lsl_supported_fraction",
    "et_raw_mapping_status", "et_raw_quality_grade",
    "et_raw_timestamp_count", "et_raw_supported_sample_count",
    "et_raw_extrapolated_sample_count",
    "et_raw_supported_fraction", "warnings",
]


def _normalise_filter(values):
    if values is None:
        return None
    return {str(value).strip() for value in values}


def _truthy(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def _validate_idun_lsl_mapping(
    mapping,
):
    """Read one usable IDUN/LSL mapping and its native raw CSV."""

    if not _truthy(mapping.get("usable", False)):
        raise ValueError("IDUN/LSL mapping is not usable")

    numeric_names = (
        "intercept_s", "scale", "idun_raw_time_origin_unix_s",
        "fit_support_start_idun_raw_s", "fit_support_end_idun_raw_s",
        "fit_support_start_lsl_s", "fit_support_end_lsl_s",
        "cap_lsl_start_s", "cap_lsl_end_s",
    )
    try:
        numeric = {name: float(mapping[name]) for name in numeric_names}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "IDUN/LSL mapping is missing numeric coefficients or support bounds"
        ) from error
    if not all(np.isfinite(value) for value in numeric.values()):
        raise ValueError("IDUN/LSL mapping contains non-finite coefficients or support")
    if numeric["scale"] <= 0:
        raise ValueError("IDUN/LSL mapping scale must be positive")
    if (numeric["fit_support_end_idun_raw_s"]
            < numeric["fit_support_start_idun_raw_s"]):
        raise ValueError("IDUN/LSL mapping native support is invalid")
    if numeric["fit_support_end_lsl_s"] < numeric["fit_support_start_lsl_s"]:
        raise ValueError("IDUN/LSL mapping LSL support is invalid")
    if numeric["cap_lsl_end_s"] < numeric["cap_lsl_start_s"]:
        raise ValueError("IDUN/LSL mapping cap-EEG support is invalid")
    expected_lsl_bounds = (
        numeric["intercept_s"]
        + numeric["scale"] * numeric["fit_support_start_idun_raw_s"],
        numeric["intercept_s"]
        + numeric["scale"] * numeric["fit_support_end_idun_raw_s"],
    )
    actual_lsl_bounds = (
        numeric["fit_support_start_lsl_s"],
        numeric["fit_support_end_lsl_s"],
    )
    if not np.allclose(
        actual_lsl_bounds, expected_lsl_bounds, rtol=1e-10, atol=1e-6,
    ):
        raise ValueError("IDUN/LSL native and LSL support bounds are inconsistent")
    if (numeric["fit_support_start_lsl_s"] < numeric["cap_lsl_start_s"] - 1e-6
            or numeric["fit_support_end_lsl_s"] > numeric["cap_lsl_end_s"] + 1e-6):
        raise ValueError("IDUN/LSL fitted support is outside cap-EEG support")

    try:
        raw_path = Path(str(mapping["raw_csv_path"])).expanduser().resolve()
    except (KeyError, OSError) as error:
        raise ValueError("IDUN/LSL mapping does not identify its raw CSV") from error

    return numeric, raw_path


def _validate_et_lsl_mapping(
    mapping,
):
    """Read one usable ET/LSL mapping."""

    if not _truthy(mapping.get("usable", False)):
        raise ValueError("ET raw/LSL mapping is not usable")

    numeric = {
        name: float(mapping[name])
        for name in (
            "intercept_s", "scale", "et_start_s", "et_end_s",
            "lsl_start_s", "lsl_end_s",
        )
    }
    if not all(np.isfinite(value) for value in numeric.values()):
        raise ValueError("ET raw/LSL mapping contains non-finite bounds or coefficients")
    if numeric["scale"] <= 0:
        raise ValueError("ET raw/LSL mapping scale must be positive")
    if numeric["et_end_s"] < numeric["et_start_s"]:
        raise ValueError("ET raw/LSL mapping ET support is invalid")
    if numeric["lsl_end_s"] < numeric["lsl_start_s"]:
        raise ValueError("ET raw/LSL mapping LSL support is invalid")

    return numeric


def discover_cap_eeg_sync_jobs(
    repair_root,
    *,
    participants=None,
    visits=None,
):
    """Build jobs from the IDUN repair manifest."""

    summary_path = Path(repair_root) / "idun_repair_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"IDUN repair summary does not exist: {summary_path}")
    summary = pd.read_csv(
        summary_path,
        dtype={
            "project": "string", "pid": "string", "visit": "string",
            "raw_csv_path": "string", "xdf_path": "string",
        },
    )
    participant_filter = _normalise_filter(participants)
    visit_filter = _normalise_filter(visits)
    if participant_filter is not None:
        summary = summary[summary["pid"].astype(str).isin(participant_filter)]
    if visit_filter is not None:
        summary = summary[summary["visit"].astype(str).isin(visit_filter)]

    rows = []
    for row in summary.to_dict("records"):
        raw_path = Path(str(row["raw_csv_path"])) if pd.notna(row["raw_csv_path"]) else None
        xdf_path = Path(str(row["xdf_path"])) if pd.notna(row["xdf_path"]) else None
        prior = pd.to_numeric(row.get("arrival_offset_reference_p01_s"), errors="coerce")
        if not _truthy(row.get("usable_relative_time")):
            status = "idun_repair_unusable"
        elif raw_path is None or not raw_path.is_file():
            status = "missing_raw"
        elif xdf_path is None or not xdf_path.is_file():
            status = "missing_xdf"
        elif not np.isfinite(prior):
            status = "missing_coarse_prior"
        else:
            status = "ready"

        rows.append({
            "project": str(row["project"]),
            "pid": str(row["pid"]),
            "visit": str(row["visit"]),
            "discovery_status": status,
            "raw_csv_path": str(raw_path) if raw_path is not None else None,
            "xdf_path": str(xdf_path) if xdf_path is not None else None,
            "coarse_prior_offset_s": float(prior) if np.isfinite(prior) else np.nan,
        })
    return pd.DataFrame(rows)


def _read_idun_raw(path: Path):
    frame = pd.read_csv(path, usecols=["timestamp", "ch1"])
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["ch1"] = pd.to_numeric(frame["ch1"], errors="coerce")
    timestamps = frame["timestamp"].to_numpy(dtype=float)
    if len(frame) < 2 or not np.all(np.isfinite(timestamps)) or not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"IDUN raw timestamps are not finite and increasing: {path}")
    return frame


def _load_cap_eeg(path: Path, stream_name="EEGfm"):
    try:
        import pyxdf
    except ImportError as error:
        raise ImportError("pyxdf is required to load cap EEG from XDF") from error

    streams, _ = pyxdf.load_xdf(
        str(path),
        select_streams=[{"name": stream_name}],
        synchronize_clocks=True,
        dejitter_timestamps=True,
        verbose=False,
    )
    if len(streams) != 1:
        raise ValueError(f"Expected one {stream_name} stream in {path}, found {len(streams)}")
    stream = streams[0]
    timestamps = np.asarray(stream["time_stamps"], dtype=float)
    values = np.asarray(stream["time_series"], dtype=float)
    if values.ndim != 2 or len(values) != len(timestamps) or len(values) < 2:
        raise ValueError(f"Invalid cap EEG stream shape in {path}: {values.shape}")
    if not np.all(np.isfinite(timestamps)) or not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"Cap EEG timestamps are not finite and increasing: {path}")

    try:
        channels = stream["info"]["desc"][0]["channels"][0]["channel"]
        labels = [channel["label"][0] for channel in channels]
    except (KeyError, IndexError, TypeError):
        labels = []
    if len(labels) != values.shape[1]:
        raise ValueError("Cap EEG channel labels do not match the data columns")
    return timestamps, values, labels


def _fill_nonfinite(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return None
    if finite.all():
        return values
    positions = np.arange(len(values), dtype=float)
    return np.interp(positions, positions[finite], values[finite])


def _moving_mean(values, width):
    width = max(1, int(width))
    return np.convolve(values, np.ones(width, dtype=float) / width, mode="same")


def _activity_envelope(values, *, base_hz, feature_hz):
    """Return a polarity-invariant, robust short-time activity envelope."""

    values = _fill_nonfinite(values)
    if values is None:
        return None
    centered = values - _moving_mean(values, round(2.0 * base_hz))
    median = float(np.median(centered))
    mad = float(np.median(np.abs(centered - median))) * 1.4826
    if np.isfinite(mad) and mad > 0:
        centered = np.clip(centered, median - 10.0 * mad, median + 10.0 * mad)
    envelope = np.sqrt(np.maximum(
        _moving_mean(centered * centered, round(0.5 * base_hz)), 0.0
    ))
    stride = int(round(base_hz / feature_hz))
    if stride < 1 or not np.isclose(base_hz / stride, feature_hz):
        raise ValueError("base_hz must be an integer multiple of feature_hz")
    return envelope[::stride]


def _robust_standardise(values):
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    scale = float(np.median(np.abs(values - median))) * 1.4826
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 0:
        return None
    return np.clip((values - median) / scale, -8.0, 8.0)


def _lagged_correlation(first, second, lag_samples):
    if lag_samples < 0:
        first = first[-lag_samples:]
        second = second[:len(second) + lag_samples]
    elif lag_samples > 0:
        first = first[:-lag_samples]
        second = second[lag_samples:]
    if len(first) < 20 or len(second) < 20:
        return np.nan
    first = _robust_standardise(first)
    second = _robust_standardise(second)
    if first is None or second is None:
        return np.nan
    first = first - np.mean(first)
    second = second - np.mean(second)
    denominator = float(np.sqrt(np.dot(first, first) * np.dot(second, second)))
    return float(np.dot(first, second) / denominator) if denominator > 0 else np.nan


def _search_lag(
    first,
    second,
    *,
    feature_hz,
    center_lag_s,
    radius_s,
    runner_up_exclusion_s=1.0,
):
    center = int(round(center_lag_s * feature_hz))
    radius = int(round(radius_s * feature_hz))
    lags = np.arange(center - radius, center + radius + 1, dtype=int)
    correlations = np.array([
        _lagged_correlation(first, second, int(lag)) for lag in lags
    ], dtype=float)
    if not np.isfinite(correlations).any():
        return None
    best_position = int(np.nanargmax(correlations))
    best_lag_samples = float(lags[best_position])
    best_correlation = float(correlations[best_position])

    # A parabolic interpolation provides sub-grid lag without changing the
    # actual correlation search or claiming more precision than QC supports.
    if 0 < best_position < len(correlations) - 1:
        left, peak, right = correlations[best_position - 1:best_position + 2]
        denominator = left - 2.0 * peak + right
        if np.all(np.isfinite([left, peak, right])) and denominator < 0:
            fraction = 0.5 * (left - right) / denominator
            if abs(fraction) <= 0.5:
                best_lag_samples += float(fraction)

    exclusion = int(round(runner_up_exclusion_s * feature_hz))
    runner_mask = np.abs(lags - lags[best_position]) > exclusion
    runner = (
        float(np.nanmax(correlations[runner_mask]))
        if runner_mask.any() and np.isfinite(correlations[runner_mask]).any()
        else np.nan
    )
    return {
        "lag_s": best_lag_samples / feature_hz,
        "peak_correlation": best_correlation,
        "runner_up_correlation": runner,
        "peak_margin": best_correlation - runner if np.isfinite(runner) else np.nan,
    }


def _robust_affine(x, y, *, minimum_points=3, minimum_span_s=600.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) == 0:
        return None
    if len(x) < minimum_points or np.ptp(x) < minimum_span_s:
        intercept = float(np.median(y - x))
        residuals = y - (intercept + x)
        return {
            "fit_status": "offset_only", "intercept_s": intercept, "scale": 1.0,
            "drift_ppm": np.nan, "inliers": np.ones(len(x), dtype=bool),
            "residuals": residuals,
        }

    inliers = np.ones(len(x), dtype=bool)
    for _ in range(6):
        scale, intercept = np.polyfit(x[inliers], y[inliers], 1)
        residuals = y - (intercept + scale * x)
        center = float(np.median(residuals[inliers]))
        mad = float(np.median(np.abs(residuals[inliers] - center))) * 1.4826
        threshold = max(0.10, 3.0 * mad)
        new_inliers = np.abs(residuals - center) <= threshold
        if new_inliers.sum() < minimum_points or np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
    scale, intercept = np.polyfit(x[inliers], y[inliers], 1)
    residuals = y - (intercept + scale * x)
    return {
        "fit_status": "affine", "intercept_s": float(intercept),
        "scale": float(scale), "drift_ppm": (float(scale) - 1.0) * 1e6,
        "inliers": inliers, "residuals": residuals,
    }


def _cap_feature(values, label_to_index, definition):
    missing = [label for label in definition if label not in label_to_index]
    if missing:
        return None
    first = values[:, label_to_index[definition[0]]]
    if len(definition) == 1:
        return first
    return first - values[:, label_to_index[definition[1]]]


def _channel_fit(
    *, project, pid, visit, cap_feature, idun_envelope, cap_envelope,
    feature_times_lsl, raw_origin_unix_s, prior_offset_s, feature_hz,
    window_s, step_s, coarse_search_radius_s, local_search_radius_s,
    minimum_peak_correlation, minimum_peak_margin,
):
    coarse = _search_lag(
        idun_envelope,
        cap_envelope,
        feature_hz=feature_hz,
        center_lag_s=0.0,
        radius_s=coarse_search_radius_s,
    )
    if coarse is None:
        return None, []

    window_samples = int(round(window_s * feature_hz))
    step_samples = int(round(step_s * feature_hz))
    rows = []
    for window_index, start in enumerate(
        range(0, max(0, len(idun_envelope) - window_samples + 1), step_samples)
    ):
        end = start + window_samples
        local = _search_lag(
            idun_envelope[start:end],
            cap_envelope[start:end],
            feature_hz=feature_hz,
            center_lag_s=coarse["lag_s"],
            radius_s=local_search_radius_s,
        )
        if local is None:
            continue
        center_position = start + window_samples // 2
        prior_lsl_center = float(feature_times_lsl[center_position])
        idun_center = prior_lsl_center - prior_offset_s - raw_origin_unix_s
        accepted = (
            local["peak_correlation"] >= minimum_peak_correlation
            and (not np.isfinite(local["peak_margin"])
                 or local["peak_margin"] >= minimum_peak_margin)
        )
        reasons = []
        if local["peak_correlation"] < minimum_peak_correlation:
            reasons.append("low_correlation")
        if np.isfinite(local["peak_margin"]) and local["peak_margin"] < minimum_peak_margin:
            reasons.append("ambiguous_peak")
        rows.append({
            "project": project, "pid": pid, "visit": visit,
            "cap_feature": cap_feature, "window_index": window_index,
            "idun_raw_center_s": idun_center,
            "prior_lsl_center_s": prior_lsl_center,
            "lag_s": local["lag_s"],
            "peak_correlation": local["peak_correlation"],
            "runner_up_correlation": local["runner_up_correlation"],
            "peak_margin": local["peak_margin"], "accepted": accepted,
            "rejection_reason": ";".join(reasons), "fit_inlier": False,
            "fit_residual_s": np.nan,
        })

    windows = pd.DataFrame(rows, columns=WINDOW_COLUMNS)
    good = windows["accepted"].astype(bool) if not windows.empty else pd.Series(dtype=bool)
    fit = None
    if good.any():
        x = windows.loc[good, "idun_raw_center_s"].to_numpy(dtype=float)
        y = (
            windows.loc[good, "prior_lsl_center_s"].to_numpy(dtype=float)
            + windows.loc[good, "lag_s"].to_numpy(dtype=float)
        )
        fit = _robust_affine(x, y)
        good_indices = windows.index[good]
        windows.loc[good_indices, "fit_inlier"] = fit["inliers"]
        windows.loc[good_indices, "fit_residual_s"] = fit["residuals"]

    if fit is None:
        channel = {
            "project": project, "pid": pid, "visit": visit,
            "cap_feature": cap_feature, "fit_status": "unavailable",
            "intercept_s": np.nan, "scale": np.nan, "drift_ppm": np.nan,
            "coarse_lag_s": coarse["lag_s"], "n_windows": len(windows),
            "n_good_windows": 0, "n_fit_inlier_windows": 0,
            "median_peak_correlation": np.nan, "median_peak_margin": np.nan,
            "fit_residual_median_s": np.nan, "fit_residual_p95_s": np.nan,
            "max_local_clock_jump_s": np.nan, "selection_score": -np.inf,
            "selected": False,
        }
        return channel, windows.to_dict("records")

    accepted = windows[windows["accepted"].astype(bool)]
    inlier_residuals = np.abs(accepted.loc[accepted["fit_inlier"].astype(bool), "fit_residual_s"])
    accepted_lags = accepted.sort_values("idun_raw_center_s")["lag_s"].to_numpy(dtype=float)
    max_jump = float(np.max(np.abs(np.diff(accepted_lags)))) if len(accepted_lags) > 1 else np.nan
    median_corr = float(accepted["peak_correlation"].median())
    residual_p95 = float(np.quantile(inlier_residuals, 0.95)) if len(inlier_residuals) else np.nan
    status_rank = 2.0 if fit["fit_status"] == "affine" else 1.0
    primary_bonus = 0.05 if cap_feature == PRIMARY_CAP_FEATURE else 0.0
    selection_score = (
        status_rank * 1000.0 + int(fit["inliers"].sum()) * 10.0
        + median_corr + primary_bonus - (residual_p95 if np.isfinite(residual_p95) else 1.0)
    )
    channel = {
        "project": project, "pid": pid, "visit": visit,
        "cap_feature": cap_feature, "fit_status": fit["fit_status"],
        "intercept_s": fit["intercept_s"], "scale": fit["scale"],
        "drift_ppm": fit["drift_ppm"], "coarse_lag_s": coarse["lag_s"],
        "n_windows": len(windows), "n_good_windows": len(accepted),
        "n_fit_inlier_windows": int(fit["inliers"].sum()),
        "median_peak_correlation": median_corr,
        "median_peak_margin": float(accepted["peak_margin"].median()),
        "fit_residual_median_s": float(np.median(inlier_residuals)) if len(inlier_residuals) else np.nan,
        "fit_residual_p95_s": residual_p95,
        "max_local_clock_jump_s": max_jump,
        "selection_score": selection_score, "selected": False,
    }
    return channel, windows.to_dict("records")


def _issue(project, pid, visit, severity, issue, detail):
    return {
        "project": project, "pid": pid, "visit": visit,
        "severity": severity, "issue": issue, "detail": detail,
    }


def _unavailable_mapping(job, status, reason):
    row = {column: np.nan for column in MAPPING_COLUMNS}
    row.update({
        "project": str(job.get("project", "ImBrAid")),
        "pid": str(job.get("pid", "")), "visit": str(job.get("visit", "")),
        "mapping_status": status, "mapping_source": "cap_eeg_content",
        "usable": False, "quality_grade": "D", "review_required": True,
        "raw_csv_path": job.get("raw_csv_path"), "xdf_path": job.get("xdf_path"),
        "mapping_reason": reason, "warnings": reason,
    })
    return row


def align_visit(
    job,
    *,
    cap_stream_name="EEGfm",
    base_hz=50.0,
    feature_hz=10.0,
    window_s=300.0,
    step_s=150.0,
    coarse_search_radius_s=30.0,
    local_search_radius_s=2.0,
    minimum_peak_correlation=0.35,
    minimum_peak_margin=0.02,
    review_drift_ppm=2000.0,
    maximum_drift_ppm=10000.0,
):
    """Align one IDUN raw recording to cap EEG and return mapping/QC tables."""

    project, pid, visit = str(job["project"]), str(job["pid"]), str(job["visit"])
    raw_path, xdf_path = Path(job["raw_csv_path"]), Path(job["xdf_path"])
    prior_offset = float(job["coarse_prior_offset_s"])
    raw = _read_idun_raw(raw_path)
    cap_timestamps, cap_values, labels = _load_cap_eeg(xdf_path, cap_stream_name)
    raw_timestamps = raw["timestamp"].to_numpy(dtype=float)
    raw_values = raw["ch1"].to_numpy(dtype=float)
    prior_lsl_timestamps = raw_timestamps + prior_offset

    overlap_start = max(float(prior_lsl_timestamps[0]), float(cap_timestamps[0])) + 3.0
    overlap_end = min(float(prior_lsl_timestamps[-1]), float(cap_timestamps[-1])) - 3.0
    if overlap_end - overlap_start < window_s:
        raise ValueError(
            f"Only {max(0.0, overlap_end - overlap_start):.1f} s of IDUN/cap EEG overlap"
        )
    base_grid = np.arange(overlap_start, overlap_end, 1.0 / base_hz)
    idun_grid = np.interp(base_grid, prior_lsl_timestamps, raw_values)
    idun_envelope = _activity_envelope(idun_grid, base_hz=base_hz, feature_hz=feature_hz)
    stride = int(round(base_hz / feature_hz))
    feature_times_lsl = base_grid[::stride][:len(idun_envelope)]

    label_to_index = {label: index for index, label in enumerate(labels)}
    channels = []
    windows = []
    for feature_name, definition in CAP_FEATURES.items():
        cap_native = _cap_feature(cap_values, label_to_index, definition)
        if cap_native is None:
            continue
        cap_grid = np.interp(base_grid, cap_timestamps, cap_native)
        cap_envelope = _activity_envelope(
            cap_grid, base_hz=base_hz, feature_hz=feature_hz
        )
        channel, channel_windows = _channel_fit(
            project=project, pid=pid, visit=visit,
            cap_feature=feature_name, idun_envelope=idun_envelope,
            cap_envelope=cap_envelope, feature_times_lsl=feature_times_lsl,
            raw_origin_unix_s=float(raw_timestamps[0]), prior_offset_s=prior_offset,
            feature_hz=feature_hz, window_s=window_s, step_s=step_s,
            coarse_search_radius_s=coarse_search_radius_s,
            local_search_radius_s=local_search_radius_s,
            minimum_peak_correlation=minimum_peak_correlation,
            minimum_peak_margin=minimum_peak_margin,
        )
        if channel is not None:
            channels.append(channel)
        windows.extend(channel_windows)

    channel_frame = pd.DataFrame(channels, columns=CHANNEL_COLUMNS)
    window_frame = pd.DataFrame(windows, columns=WINDOW_COLUMNS)
    usable_channels = channel_frame[
        channel_frame["fit_status"].isin(["affine", "offset_only"])
    ].copy()
    if usable_channels.empty:
        raise ValueError("No cap EEG feature produced a usable content mapping")
    usable_channels.sort_values("selection_score", ascending=False, inplace=True)
    selected_index = usable_channels.index[0]
    channel_frame.loc[selected_index, "selected"] = True
    selected = channel_frame.loc[selected_index]

    midpoint = float(raw_timestamps[-1] - raw_timestamps[0]) / 2.0
    comparable = channel_frame[
        channel_frame["fit_status"].eq("affine")
        & (channel_frame["n_fit_inlier_windows"] >= 5)
        & (channel_frame.index != selected_index)
    ]
    if comparable.empty:
        max_offset_difference = max_drift_difference = np.nan
        support_count = 1
        agreement = False
    else:
        selected_at_mid = float(selected["intercept_s"] + selected["scale"] * midpoint)
        other_at_mid = (
            comparable["intercept_s"].to_numpy(dtype=float)
            + comparable["scale"].to_numpy(dtype=float) * midpoint
        )
        offset_differences = np.abs(other_at_mid - selected_at_mid)
        selected_drift = float(selected["drift_ppm"]) if np.isfinite(selected["drift_ppm"]) else 0.0
        other_drift = comparable["drift_ppm"].to_numpy(dtype=float)
        drift_differences = np.abs(other_drift - selected_drift)
        max_offset_difference = float(np.max(offset_differences))
        max_drift_difference = float(np.max(drift_differences))
        support_count = len(comparable) + 1
        agreement = max_offset_difference <= 0.5 and max_drift_difference <= 1000.0

    fit_status = str(selected["fit_status"])
    drift = float(selected["drift_ppm"]) if np.isfinite(selected["drift_ppm"]) else np.nan
    residual_p95 = float(selected["fit_residual_p95_s"])
    median_corr = float(selected["median_peak_correlation"])
    n_inliers = int(selected["n_fit_inlier_windows"])
    selected_windows = window_frame[
        window_frame["cap_feature"] == selected["cap_feature"]
    ].sort_values("idun_raw_center_s")
    accepted_lags = selected_windows.loc[
        selected_windows["accepted"].astype(bool), "lag_s"
    ].to_numpy(dtype=float)
    warnings = []
    if fit_status == "offset_only":
        warnings.append("too few supported windows to estimate clock drift")
    if n_inliers < 5:
        warnings.append(f"only {n_inliers} fit-inlier windows were available")
    if median_corr < 0.60:
        warnings.append(f"median peak correlation was {median_corr:.3f}")
    if residual_p95 > 0.20:
        warnings.append(f"fit residual P95 was {residual_p95:.3f} s")
    if np.isfinite(drift) and abs(drift) > review_drift_ppm:
        warnings.append(f"absolute clock drift was {abs(drift):.1f} ppm")
    if not agreement:
        warnings.append("cap EEG feature mappings require review")
    max_clock_jump = (
        float(np.max(np.abs(np.diff(accepted_lags)))) if len(accepted_lags) > 1 else np.nan
    )
    if np.isfinite(max_clock_jump) and max_clock_jump > 0.5:
        warnings.append(f"maximum adjacent local-lag change was {max_clock_jump:.3f} s")

    if fit_status == "affine" and np.isfinite(drift) and abs(drift) > maximum_drift_ppm:
        mapping_status, usable, grade, review = "invalid_affine", False, "D", True
        reason = "Estimated clock drift exceeded the configured maximum"
    elif (fit_status == "affine" and n_inliers >= 5 and median_corr >= 0.60
          and residual_p95 <= 0.20 and agreement
          and (not np.isfinite(max_clock_jump) or max_clock_jump <= 0.5)):
        mapping_status, usable, grade, review = "affine", True, "A", False
        reason = "Supported affine mapping from IDUN and cap EEG content"
    elif fit_status == "affine":
        mapping_status, usable, grade, review = "affine_review", True, "B", True
        reason = "Usable affine mapping with review-level quality metrics"
    else:
        mapping_status, usable, grade, review = "offset_only", True, "C", True
        reason = "Usable offset-only mapping; clock drift was not estimated"

    intercept, scale = float(selected["intercept_s"]), float(selected["scale"])
    raw_duration = float(raw_timestamps[-1] - raw_timestamps[0])
    supported_windows = selected_windows[
        selected_windows["accepted"].astype(bool)
        & selected_windows["fit_inlier"].astype(bool)
    ]
    support_start = max(
        0.0, float(supported_windows["idun_raw_center_s"].min()) - window_s / 2.0
    )
    support_end = min(
        raw_duration, float(supported_windows["idun_raw_center_s"].max()) + window_s / 2.0
    )
    mapping = {
        "project": project, "pid": pid, "visit": visit,
        "mapping_status": mapping_status, "mapping_source": "cap_eeg_content",
        "usable": usable, "quality_grade": grade, "review_required": review,
        "selected_cap_feature": selected["cap_feature"],
        "intercept_s": intercept, "scale": scale, "drift_ppm": drift,
        "idun_raw_time_origin_unix_s": float(raw_timestamps[0]),
        "idun_raw_duration_s": raw_duration,
        "fit_support_start_idun_raw_s": support_start,
        "fit_support_end_idun_raw_s": support_end,
        "fit_support_start_lsl_s": intercept + scale * support_start,
        "fit_support_end_lsl_s": intercept + scale * support_end,
        "cap_lsl_start_s": float(cap_timestamps[0]),
        "cap_lsl_end_s": float(cap_timestamps[-1]),
        "overlap_duration_s": float(overlap_end - overlap_start),
        "coarse_prior_offset_s": prior_offset,
        "coarse_prior_source": "idun_lsl_arrival_p01_search_prior_only",
        "coarse_lag_s": float(selected["coarse_lag_s"]),
        "n_windows": int(selected["n_windows"]),
        "n_good_windows": int(selected["n_good_windows"]),
        "n_fit_inlier_windows": n_inliers,
        "median_peak_correlation": median_corr,
        "median_peak_margin": float(selected["median_peak_margin"]),
        "fit_residual_median_s": float(selected["fit_residual_median_s"]),
        "fit_residual_p95_s": residual_p95,
        "max_local_clock_jump_s": max_clock_jump,
        "support_channel_count": support_count,
        "max_channel_offset_difference_s": max_offset_difference,
        "max_channel_drift_difference_ppm": max_drift_difference,
        "channel_agreement": agreement,
        "raw_csv_path": str(raw_path), "xdf_path": str(xdf_path),
        "mapping_reason": reason, "warnings": "; ".join(warnings),
    }

    issues = []
    if review:
        issues.append(_issue(project, pid, visit, "warning",
                             "idun_cap_mapping_review", "; ".join(warnings) or reason))
    if not usable:
        issues.append(_issue(project, pid, visit, "error",
                             "idun_cap_mapping_unusable", reason))
    return mapping, channel_frame, window_frame, issues


def build_idun_cap_alignments(jobs, **kwargs):
    """Run all jobs and retain unavailable visits in the mapping table."""

    mappings, channels, windows, issues = [], [], [], []
    for job in jobs.to_dict("records"):
        project, pid, visit = str(job.get("project")), str(job.get("pid")), str(job.get("visit"))
        if job.get("discovery_status") != "ready":
            reason = f"Discovery status: {job.get('discovery_status')}"
            mappings.append(_unavailable_mapping(job, "unavailable", reason))
            issues.append(_issue(project, pid, visit, "error",
                                 "idun_cap_source_unavailable", reason))
            continue
        try:
            mapping, visit_channels, visit_windows, visit_issues = align_visit(
                job, **kwargs
            )
            mappings.append(mapping)
            channels.append(visit_channels)
            windows.append(visit_windows)
            issues.extend(visit_issues)
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            mappings.append(_unavailable_mapping(job, "failed", reason))
            issues.append(_issue(project, pid, visit, "error",
                                 "idun_cap_alignment_failed", reason))

    mapping_frame = pd.DataFrame(mappings, columns=MAPPING_COLUMNS)
    channel_frame = (
        pd.concat(channels, ignore_index=True) if channels
        else pd.DataFrame(columns=CHANNEL_COLUMNS)
    )
    window_frame = (
        pd.concat(windows, ignore_index=True) if windows
        else pd.DataFrame(columns=WINDOW_COLUMNS)
    )
    issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    return mapping_frame, channel_frame, window_frame, issue_frame


def _to_csv(frame, path, **to_csv_kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, **to_csv_kwargs)


def save_idun_cap_alignment_results(
    mappings,
    channels,
    windows,
    issues,
    *,
    alignment_csv,
    channels_csv,
    windows_csv,
    issues_csv,
):
    paths = [Path(alignment_csv), Path(channels_csv), Path(windows_csv), Path(issues_csv)]
    frames = [mappings, channels, windows, issues]
    for frame, path in zip(frames, paths):
        _to_csv(frame, path)
    return tuple(paths)


def export_synchronized_idun_data(
    mappings,
    *,
    output_dir,
    et_lsl_alignment_csv=None,
    overwrite=False,
):
    """Write IDUN timestamps using accepted mappings, including extrapolation."""

    mappings = pd.DataFrame(mappings).copy()
    if mappings.duplicated(["pid", "visit"]).any():
        raise ValueError("IDUN/cap EEG mappings contain duplicate pid/visit rows")
    et_table = None
    et_source_warning = ""
    et_source_status = ""
    if et_lsl_alignment_csv is not None:
        et_path = Path(et_lsl_alignment_csv)
        if et_path.is_file():
            try:
                et_table = pd.read_csv(
                    et_path,
                    dtype=str,
                    keep_default_na=False,
                )
            except (OSError, UnicodeError, ValueError) as error:
                et_source_status = "alignment_csv_unreadable"
                et_source_warning = (
                    f"ET raw/LSL alignment CSV could not be read: {et_path}; "
                    f"{type(error).__name__}: {error}"
                )
        else:
            et_source_status = "alignment_csv_missing"
            et_source_warning = f"ET raw/LSL alignment CSV not found: {et_path}"

    output_dir = Path(output_dir)
    summaries, issues = [], []
    for mapping in mappings.to_dict("records"):
        project = str(mapping.get("project", "ImBrAid"))
        pid, visit = str(mapping.get("pid", "")), str(mapping.get("visit", ""))
        warnings = []
        summary = {
            column: np.nan for column in SYNCHRONIZED_SUMMARY_COLUMNS
        }
        summary.update({
            "project": project, "pid": pid, "visit": visit,
            "export_status": "not_written",
            "idun_mapping_status": mapping.get("mapping_status"),
            "idun_quality_grade": mapping.get("quality_grade"),
            "source_raw_csv_path": mapping.get("raw_csv_path"),
            "et_raw_mapping_status": "not_requested",
            "warnings": "",
        })
        if not _truthy(mapping.get("usable", False)):
            mapping_status = str(mapping.get("mapping_status", "")).strip()
            detail = "IDUN/LSL mapping is unavailable"
            if mapping_status:
                detail = f"{detail}: {mapping_status}"
            summary.update({
                "export_status": "skipped_unavailable",
                "warnings": detail,
            })
            issues.append(_issue(
                project, pid, visit, "warning",
                "idun_mapping_unavailable", detail,
            ))
            summaries.append(summary)
            continue
        try:
            idun_values, raw_path = _validate_idun_lsl_mapping(mapping)
            raw = _read_idun_raw(raw_path)
            raw_timestamps = raw["timestamp"].to_numpy(dtype=float)
            raw_relative = (
                raw_timestamps - idun_values["idun_raw_time_origin_unix_s"]
            )
            lsl_timestamps = map_idun_raw_to_lsl(
                mapping, raw_relative, allow_outside_support=True
            )
            lsl_finite = np.isfinite(lsl_timestamps)
            lsl_supported = (
                lsl_finite
                & (raw_relative >= idun_values["fit_support_start_idun_raw_s"])
                & (raw_relative <= idun_values["fit_support_end_idun_raw_s"])
            )

            et_timestamps = np.full(len(raw), np.nan, dtype=float)
            et_supported = np.zeros(len(raw), dtype=bool)
            if et_source_warning:
                summary["et_raw_mapping_status"] = et_source_status
                warnings.append(et_source_warning)
            elif et_table is None:
                summary["et_raw_mapping_status"] = "not_requested"
                warnings.append(
                    "ET raw/LSL alignment CSV was not provided; "
                    "et_raw_timestamp is unavailable"
                )
            else:
                try:
                    selected_et = et_table[
                        (et_table["project"].astype(str) == project)
                        & (et_table["pid"].astype(str) == pid)
                        & (et_table["visit"].astype(str) == visit)
                    ]
                    if len(selected_et) != 1:
                        summary["et_raw_mapping_status"] = "mapping_not_unique"
                        raise ValueError(
                            "expected one ET raw/LSL mapping, "
                            f"found {len(selected_et)}"
                        )

                    et_mapping = selected_et.iloc[0]
                    source_et_status = str(
                        et_mapping.get("mapping_status", "")
                    ).strip()
                    summary["et_raw_mapping_status"] = source_et_status
                    summary["et_raw_quality_grade"] = et_mapping.get("quality_grade")
                    et_values = _validate_et_lsl_mapping(et_mapping)
                    candidate_et = (
                        lsl_timestamps - et_values["intercept_s"]
                    ) / et_values["scale"]
                    et_supported = (
                        lsl_supported
                        & np.isfinite(candidate_et)
                        & (candidate_et >= et_values["et_start_s"])
                        & (candidate_et <= et_values["et_end_s"])
                        & (lsl_timestamps >= et_values["lsl_start_s"])
                        & (lsl_timestamps <= et_values["lsl_end_s"])
                    )
                    et_finite = lsl_finite & np.isfinite(candidate_et)
                    et_timestamps[et_finite] = candidate_et[et_finite]
                except (KeyError, OSError, TypeError, ValueError) as error:
                    source_status = str(
                        summary["et_raw_mapping_status"]
                    ).strip()
                    if source_status in {
                        "content_affine", "content_review", "legacy_fallback",
                    }:
                        summary["et_raw_mapping_status"] = "validation_failed"
                    elif not source_status or source_status == "not_requested":
                        summary["et_raw_mapping_status"] = "validation_failed"
                    warnings.append(
                        "ET raw/LSL mapping was not used; "
                        f"et_raw_timestamp is unavailable: "
                        f"{type(error).__name__}: {error}"
                    )

            synchronized = pd.DataFrame(
                {
                    "sample_index": np.arange(len(raw), dtype=np.int64),
                    "idun_timestamp": raw_timestamps,
                    "et_raw_timestamp": et_timestamps,
                    "lsl_timestamp": np.where(
                        lsl_finite, lsl_timestamps, np.nan
                    ),
                    "eeg_ch1": raw["ch1"].to_numpy(dtype=float),
                },
                columns=PUBLICATION_DATA_COLUMNS,
            )
            output_path = output_dir / f"{pid}_{visit}_idun_synchronized.csv"
            sample_count = len(synchronized)
            lsl_timestamp_count = int(lsl_finite.sum())
            lsl_supported_count = int(lsl_supported.sum())
            et_timestamp_count = int(np.isfinite(et_timestamps).sum())
            et_supported_count = int(et_supported.sum())
            if sample_count <= 0:
                raise ValueError("IDUN publication source contains no samples")
            if lsl_timestamp_count <= 0:
                raise ValueError(
                    "IDUN publication source has no finite mapped LSL timestamps"
                )

            export_status = "written"
            if output_path.is_file() and not overwrite:
                export_status = "skipped_existing"
            else:
                _to_csv(synchronized, output_path, float_format="%.9f")

            summary.update({
                "export_status": export_status,
                "synchronized_csv_path": str(output_path),
                "sample_count": sample_count,
                "lsl_timestamp_count": lsl_timestamp_count,
                "lsl_supported_sample_count": lsl_supported_count,
                "lsl_extrapolated_sample_count": lsl_timestamp_count - lsl_supported_count,
                "lsl_supported_fraction": lsl_supported_count / sample_count,
                "et_raw_timestamp_count": et_timestamp_count,
                "et_raw_supported_sample_count": et_supported_count,
                "et_raw_extrapolated_sample_count": et_timestamp_count - et_supported_count,
                "et_raw_supported_fraction": et_supported_count / sample_count,
                "warnings": "; ".join(warnings),
            })
            if warnings:
                issues.append(_issue(
                    project, pid, visit, "warning",
                    "synchronized_timestamp_warning", "; ".join(warnings),
                ))
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            summary.update({"export_status": "failed", "warnings": detail})
            issues.append(_issue(
                project, pid, visit, "error",
                "synchronized_idun_export_failed", detail,
            ))
        summaries.append(summary)

    summary_frame = pd.DataFrame(summaries, columns=SYNCHRONIZED_SUMMARY_COLUMNS)
    issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    return summary_frame, issue_frame


def save_idun_export_summary(summary, path):
    """Save the completed Step-06 IDUN export summary."""

    frame = pd.DataFrame(summary).reindex(columns=SYNCHRONIZED_SUMMARY_COLUMNS)
    _to_csv(frame, path)
    return Path(path).expanduser().resolve()


def save_idun_export_issues(issues, path):
    """Save issues returned by the synchronized IDUN exporter."""

    frame = pd.DataFrame(issues).reindex(columns=ISSUE_COLUMNS)
    _to_csv(frame, path)
    return Path(path).expanduser().resolve()


def map_idun_raw_to_lsl(
    mapping_row,
    idun_raw_relative_s,
    *,
    allow_outside_support=True,
):
    """Apply one accepted mapping, extrapolating beyond fitted support by default."""

    if isinstance(mapping_row, pd.DataFrame):
        if len(mapping_row) != 1:
            raise ValueError(f"Expected one mapping row, found {len(mapping_row)}")
        mapping_row = mapping_row.iloc[0]
    if not _truthy(mapping_row.get("usable", False)):
        raise ValueError("The selected IDUN/cap EEG mapping is not usable")
    intercept = float(mapping_row["intercept_s"])
    scale = float(mapping_row["scale"])
    if not np.isfinite(intercept) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Mapping coefficients must be finite and scale must be positive")
    times = np.asarray(idun_raw_relative_s, dtype=float)
    support_start = float(mapping_row["fit_support_start_idun_raw_s"])
    support_end = float(mapping_row["fit_support_end_idun_raw_s"])
    if (not allow_outside_support and np.isfinite(times).any()
            and (np.nanmin(times) < support_start or np.nanmax(times) > support_end)):
        raise ValueError(
            f"IDUN raw time is outside fitted support [{support_start:.3f}, {support_end:.3f}] s"
        )
    return intercept + scale * times


__all__ = [
    "CHANNEL_COLUMNS",
    "ISSUE_COLUMNS",
    "MAPPING_COLUMNS",
    "PUBLICATION_DATA_COLUMNS",
    "SYNCHRONIZED_SUMMARY_COLUMNS",
    "WINDOW_COLUMNS",
    "build_idun_cap_alignments",
    "discover_cap_eeg_sync_jobs",
    "export_synchronized_idun_data",
    "map_idun_raw_to_lsl",
    "save_idun_cap_alignment_results",
    "save_idun_export_issues",
    "save_idun_export_summary",
]
