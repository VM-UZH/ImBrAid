import numpy as np
from scipy import signal as sc_signal

from helpers.project_utils.helper_utils import ET_FREQ

# Both recordings are resampled onto this grid before correlating.
DEFAULT_HZ = ET_FREQ

# Two groups count as agreeing when their offsets are closer than this.
INLIER_THRESHOLD_S = 0.5


# Channels are grouped per physical quantity: only axes of the same sensor
# form one vector.
IMU_GROUPS = {
    "magnetometer": ["magnetometer.x", "magnetometer.y", "magnetometer.z"],
    "accelerometer": ["accelerometer.x", "accelerometer.y", "accelerometer.z"],
    "gyro": ["gyro.x", "gyro.y", "gyro.z"],
}
ET_GROUPS = {
    "gaze3d": ["gaze3d.x", "gaze3d.y", "gaze3d.z"],
    "gaze2d": ["gaze2d.x", "gaze2d.y"],
    # Pupil diameter is a scalar, so it forms a group of its own.
    "left-pupil": ["left-pupil"],
    "right-pupil": ["right-pupil"],
    # Direction vectors are grouped per eye.
    "left-gaze-direction": ["left-gaze-direction.x", "left-gaze-direction.y",
                            "left-gaze-direction.z"],
    "right-gaze-direction": ["right-gaze-direction.x", "right-gaze-direction.y",
                             "right-gaze-direction.z"],
}

MIN_SAMPLES = 10

# Native ET IMU in the inspected recordings is close to 120 Hz for gyro and
# accelerometer, while the magnetometer is close to 10 Hz.  Keeping those
# rates avoids throwing away the distinctive motion edges that locate a lag.
GROUP_SAMPLE_HZ = {
    "gyro": 120.0,
    "accelerometer": 120.0,
    "magnetometer": 10.0,
    "gaze3d": 50.0,
    "gaze2d": 50.0,
    "left-pupil": 50.0,
    "right-pupil": 50.0,
    "left-gaze-direction": 50.0,
    "right-gaze-direction": 50.0,
}

WINDOW_LENGTH_S = 60.0
WINDOW_STEP_S = 30.0
LOCAL_SEARCH_RADIUS_S = 2.0
MAX_INTERPOLATION_GAP_S = 0.200
MIN_WINDOW_OVERLAP_S = 10.0
MIN_PEAK_CORRELATION = 0.30
MIN_PEAK_MARGIN = 0.02
LOCAL_GROUP_INLIER_THRESHOLD_S = 0.150


def _resample_channels(et_df, lsl_df, features, sample_hz=DEFAULT_HZ):
    """
    Resample one group of channels onto a regular ET grid and a regular LSL grid.

    Returns:
        (et_matrix, lsl_matrix, t_et0, t_lsl0) with both matrices of shape
        (n_channels, n_samples) and every channel normalised to zero mean and
        unit standard deviation, or None when either side has fewer than
        MIN_SAMPLES samples for this group.
    """
    et_sub = et_df[['timestamp'] + list(features)].dropna()
    lsl_sub = lsl_df[list(features)].dropna()
    if len(et_sub) < MIN_SAMPLES or len(lsl_sub) < MIN_SAMPLES:
        return None

    # All axes of one sensor share the same timestamps, so one grid per side.
    t_et = et_sub['timestamp'].values
    t_lsl = lsl_sub.index.view('int64') / 1e9

    et_grid = np.arange(t_et[0], t_et[-1], 1.0 / sample_hz)
    lsl_grid = np.arange(t_lsl[0], t_lsl[-1], 1.0 / sample_hz)

    et_channels, lsl_channels = [], []
    for feature in features:
        et_values = np.interp(et_grid, t_et, et_sub[feature].values.astype(float))
        lsl_values = np.interp(lsl_grid, t_lsl, lsl_sub[feature].values.astype(float))
        et_values = (et_values - et_values.mean()) / (et_values.std() + 1e-10)
        lsl_values = (lsl_values - lsl_values.mean()) / (lsl_values.std() + 1e-10)
        et_channels.append(et_values)
        lsl_channels.append(lsl_values)

    return np.array(et_channels), np.array(lsl_channels), t_et[0], t_lsl[0]


def _align_vector(et_df, lsl_df, features, sample_hz=DEFAULT_HZ):
    """
    Cross-correlate one group of channels as a vector and return its offset.

    The vector cross-correlation is the sum of the per-axis correlations; its
    peak is the time shift. Summing is only valid because every axis of a
    group shares one grid per side, so all axes produce the same lag axis.
    Returns NaN when the group has too little data.
    """
    packed = _resample_channels(et_df, lsl_df, features, sample_hz)
    if packed is None:
        return np.nan
    et_matrix, lsl_matrix, t_et0, t_lsl0 = packed

    corr_sum = None
    for et_values, lsl_values in zip(et_matrix, lsl_matrix):
        corr = sc_signal.correlate(et_values, lsl_values, mode='full', method='fft')
        corr_sum = corr if corr_sum is None else corr_sum + corr

    lags = sc_signal.correlation_lags(et_matrix.shape[1], lsl_matrix.shape[1], mode='full')
    peak_lag = lags[np.argmax(corr_sum)]

    return t_lsl0 - t_et0 - peak_lag / sample_hz


def offset_candidates(et_df, lsl_df, sample_hz=DEFAULT_HZ,
                      inlier_threshold=INLIER_THRESHOLD_S, verbose=True):
    """
    Rank every offset the channel groups support, best first.

    Each group votes with one offset. Groups whose offsets agree within
    `inlier_threshold` form a candidate, and a candidate is described by the
    median of its members. Candidates are ranked by how many groups back them
    and then by how tightly those agree, so a value three sensors arrive at
    independently outranks one that a single sensor produced.

    When the sensors genuinely disagree - the case that used to be reported as
    a perfect fit, because the standard deviation of one surviving group is
    zero - this returns each of them separately, so the caller can try the
    next one instead of trusting the first.

    Args:
        et_df: IMU or gaze DataFrame from `et_loaders.load_imu` / `load_et`.
        lsl_df: the LSL ET stream from `lsl_loaders.load_lsl_for_visit`.
        sample_hz: resampling rate used for the correlation.
        inlier_threshold: how close two groups have to be to count as agreeing.
        verbose: print the per-group offsets.

    Returns:
        List of (offset_s, std_s, n_groups, group_names), best first. Empty
        when no group had enough data.
    """
    columns = set(et_df.columns)
    all_groups = {**IMU_GROUPS, **ET_GROUPS}
    active = {name: group for name, group in all_groups.items()
              if all(channel in columns for channel in group)}

    offsets = {name: _align_vector(et_df, lsl_df, group, sample_hz)
               for name, group in active.items()}
    values = {name: value for name, value in offsets.items() if not np.isnan(value)}

    if verbose:
        print(f"{len(offsets)} groups: {offsets}")

    if not values:
        return []

    # One candidate per group of mutually agreeing offsets; identical member
    # sets collapse onto each other.
    clusters = {}
    for seed in values.values():
        members = tuple(sorted(name for name, value in values.items()
                               if abs(value - seed) < inlier_threshold))
        if members not in clusters:
            member_values = np.array([values[name] for name in members])
            clusters[members] = (float(np.median(member_values)),
                                 float(np.std(member_values)))

    candidates = [(offset, std, len(members), list(members))
                  for members, (offset, std) in clusters.items()]
    # Most supporting groups first, then the tightest, then by value so the
    # order never depends on dictionary iteration.
    candidates.sort(key=lambda c: (-c[2], c[1], c[0]))

    if verbose and len(candidates) > 1:
        print(f"{len(candidates)} candidate offsets: "
              + ", ".join(f"{o:.3f}s ({n} group{'s' if n > 1 else ''})"
                          for o, _, n, _ in candidates))

    return candidates


def _interpolate_without_long_gaps(times, values, grid, max_gap_s):
    """Interpolate samples while masking grid points that bridge a long gap."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(times)
    times, values = times[order], values[order]

    # Repeated timestamps make the left/right-neighbour test ambiguous. Keep
    # the last sample at each time, matching np.interp's effective behaviour.
    unique_times, unique_indices = np.unique(times[::-1], return_index=True)
    keep = len(times) - 1 - unique_indices
    order = np.argsort(unique_times)
    times, values = unique_times[order], values[keep[order]]
    if len(times) < 2:
        return np.zeros_like(grid, dtype=float), np.zeros_like(grid, dtype=bool)

    right = np.searchsorted(times, grid, side="left")
    inside = (right > 0) & (right < len(times))
    left_index = np.clip(right - 1, 0, len(times) - 1)
    right_index = np.clip(right, 0, len(times) - 1)
    gaps = times[right_index] - times[left_index]
    valid = inside & (gaps <= max_gap_s)

    interpolated = np.interp(grid, times, values)
    interpolated[~valid] = 0.0
    return interpolated, valid


def _normalise_window(values, valid):
    """Remove a linear trend and robustly scale one interpolated channel."""
    output = np.zeros_like(values, dtype=float)
    if valid.sum() < MIN_SAMPLES:
        return output, np.zeros_like(valid, dtype=bool)

    x = np.flatnonzero(valid).astype(float)
    y = values[valid]
    slope, intercept = np.polyfit(x, y, 1)
    detrended = y - (intercept + slope * x)
    center = float(np.median(detrended))
    mad = float(np.median(np.abs(detrended - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(detrended))
    if not np.isfinite(scale) or scale < 1e-8:
        return output, np.zeros_like(valid, dtype=bool)

    output[valid] = (detrended - center) / scale
    return output, valid


def _masked_normalised_vector_correlation(et_matrix, lsl_matrix,
                                           et_valid, lsl_valid):
    """Calculate per-lag Pearson correlation without filling invalid gaps."""
    n_et = et_matrix.shape[1]
    n_lsl = lsl_matrix.shape[1]
    lags = sc_signal.correlation_lags(n_et, n_lsl, mode="full")
    covariance = np.zeros(len(lags), dtype=float)
    variance_et = np.zeros(len(lags), dtype=float)
    variance_lsl = np.zeros(len(lags), dtype=float)
    overlap = np.zeros(len(lags), dtype=float)

    for et_values, lsl_values, et_mask, lsl_mask in zip(
            et_matrix, lsl_matrix, et_valid, lsl_valid):
        mx = et_mask.astype(float)
        my = lsl_mask.astype(float)
        x = et_values * mx
        y = lsl_values * my

        count = sc_signal.correlate(mx, my, mode="full", method="fft")
        sum_xy = sc_signal.correlate(x, y, mode="full", method="fft")
        sum_x = sc_signal.correlate(x, my, mode="full", method="fft")
        sum_y = sc_signal.correlate(mx, y, mode="full", method="fft")
        sum_x2 = sc_signal.correlate(x * x, my, mode="full", method="fft")
        sum_y2 = sc_signal.correlate(mx, y * y, mode="full", method="fft")

        safe_count = np.maximum(count, 1.0)
        covariance += sum_xy - sum_x * sum_y / safe_count
        variance_et += np.maximum(sum_x2 - sum_x * sum_x / safe_count, 0.0)
        variance_lsl += np.maximum(sum_y2 - sum_y * sum_y / safe_count, 0.0)
        overlap += np.maximum(count, 0.0)

    denominator = np.sqrt(variance_et * variance_lsl)
    correlation = np.full(len(lags), np.nan, dtype=float)
    usable = denominator > 1e-12
    correlation[usable] = covariance[usable] / denominator[usable]
    correlation = np.clip(correlation, -1.0, 1.0)
    return lags, correlation, overlap


def _local_group_offset(et_df, lsl_df, features, group_name, center_s,
                        prior_offset_s, window_s, search_radius_s,
                        max_gap_s, sample_hz_by_group=None):
    """Estimate one sensor group's offset in one local time window."""
    sample_hz = (
        sample_hz_by_group.get(group_name)
        if sample_hz_by_group and group_name in sample_hz_by_group
        else GROUP_SAMPLE_HZ.get(group_name, DEFAULT_HZ)
    )
    sample_hz = float(sample_hz)
    if not np.isfinite(sample_hz) or sample_hz <= 0:
        return None
    effective_max_gap_s = max(float(max_gap_s), 2.5 / sample_hz)
    half_window = window_s / 2.0
    grid = np.arange(center_s - half_window, center_s + half_window,
                     1.0 / sample_hz)
    if len(grid) < MIN_SAMPLES:
        return None

    et_sub = et_df[["timestamp"] + list(features)].dropna()
    lsl_sub = lsl_df[list(features)].dropna()
    if len(et_sub) < MIN_SAMPLES or len(lsl_sub) < MIN_SAMPLES:
        return None

    et_times = et_sub["timestamp"].to_numpy(dtype=float)
    # Move LSL timestamps onto the approximate ET grid. Correlation estimates
    # the small residual around this prior rather than searching the full visit.
    lsl_times = lsl_sub.index.view("int64") / 1e9 - float(prior_offset_s)

    et_channels, lsl_channels = [], []
    et_masks, lsl_masks = [], []
    for feature in features:
        et_values, et_valid = _interpolate_without_long_gaps(
            et_times, et_sub[feature].to_numpy(dtype=float), grid,
            effective_max_gap_s)
        lsl_values, lsl_valid = _interpolate_without_long_gaps(
            lsl_times, lsl_sub[feature].to_numpy(dtype=float), grid,
            effective_max_gap_s)
        et_values, et_valid = _normalise_window(et_values, et_valid)
        lsl_values, lsl_valid = _normalise_window(lsl_values, lsl_valid)
        et_channels.append(et_values)
        lsl_channels.append(lsl_values)
        et_masks.append(et_valid)
        lsl_masks.append(lsl_valid)

    et_matrix = np.asarray(et_channels)
    lsl_matrix = np.asarray(lsl_channels)
    et_valid = np.asarray(et_masks)
    lsl_valid = np.asarray(lsl_masks)
    lags, correlation, overlap = _masked_normalised_vector_correlation(
        et_matrix, lsl_matrix, et_valid, lsl_valid)

    radius_samples = max(1, int(round(search_radius_s * sample_hz)))
    search = (np.abs(lags) <= radius_samples) & np.isfinite(correlation)
    min_overlap_values = MIN_WINDOW_OVERLAP_S * sample_hz * len(features)
    search &= overlap >= min_overlap_values
    indices = np.flatnonzero(search)
    if not len(indices):
        return None

    peak_index = indices[np.argmax(correlation[indices])]
    peak = float(correlation[peak_index])
    peak_lag = float(lags[peak_index])

    # Quadratic interpolation locates the peak between resampling-grid points.
    if 0 < peak_index < len(correlation) - 1:
        left, middle, right = correlation[peak_index - 1:peak_index + 2]
        denominator = left - 2.0 * middle + right
        if np.all(np.isfinite([left, middle, right])) and abs(denominator) > 1e-12:
            fraction = 0.5 * (left - right) / denominator
            if abs(fraction) <= 1.0:
                peak_lag += float(fraction)

    exclusion = max(1, int(round(0.25 * sample_hz)))
    alternatives = indices[np.abs(indices - peak_index) > exclusion]
    second_peak = (float(np.max(correlation[alternatives]))
                   if len(alternatives) else -1.0)
    peak_margin = peak - second_peak
    corrected_offset = float(prior_offset_s) - peak_lag / sample_hz
    distance_from_prior = abs(corrected_offset - float(prior_offset_s))
    boundary_margin_s = max(0.25, 2.0 / sample_hz)

    return {
        "center_et_s": float(center_s),
        "group": group_name,
        "offset_s": corrected_offset,
        "peak_correlation": peak,
        "peak_margin": float(peak_margin),
        "overlap_s": float(overlap[peak_index] / (sample_hz * len(features))),
        "sample_hz": float(sample_hz),
        "distance_from_prior_s": float(distance_from_prior),
        "at_search_boundary": bool(
            distance_from_prior >= max(0.0, search_radius_s - boundary_margin_s)),
    }


def _fit_window_offsets(centers, offsets, max_abs_drift_ppm=2_000.0):
    """Robustly fit a clock offset that changes linearly over ET time."""
    x = np.asarray(centers, dtype=float)
    y = np.asarray(offsets, dtype=float)
    inliers = np.ones(len(x), dtype=bool)

    if len(x) < 3 or np.ptp(x) < WINDOW_LENGTH_S:
        intercept = float(np.median(y))
        slope = 0.0
    else:
        for _ in range(5):
            slope, intercept = np.polyfit(x[inliers], y[inliers], 1)
            residuals = y - (intercept + slope * x)
            center = float(np.median(residuals[inliers]))
            mad = float(np.median(np.abs(residuals[inliers] - center)))
            threshold = max(0.040, 3.0 * 1.4826 * mad)
            new_inliers = np.abs(residuals - center) <= threshold
            if new_inliers.sum() < 3 or np.array_equal(new_inliers, inliers):
                break
            inliers = new_inliers
        slope, intercept = np.polyfit(x[inliers], y[inliers], 1)

    raw_drift_ppm = float(slope * 1e6)
    drift_was_clamped = False
    if (max_abs_drift_ppm is not None
            and abs(raw_drift_ppm) > float(max_abs_drift_ppm)):
        # An extreme slope is almost always a sequence mismatch or timestamp
        # discontinuity. Keep a constant model so it cannot warp the visit.
        slope = 0.0
        intercept = float(np.median(y[inliers]))
        drift_was_clamped = True

    residuals = y - (intercept + slope * x)
    absolute = np.abs(residuals[inliers])
    return {
        "intercept_s": float(intercept),
        "scale": float(1.0 + slope),
        "drift_ppm": float(slope * 1e6),
        "raw_drift_ppm": raw_drift_ppm,
        "drift_was_clamped": drift_was_clamped,
        "fit_inliers": int(inliers.sum()),
        "fit_residual_mad_s": float(np.median(absolute)),
        "fit_residual_p95_s": float(np.percentile(absolute, 95)),
    }


def _windowed_alignment_for_prior(et_df, lsl_df, prior_offset_s,
                                  window_s, step_s, search_radius_s,
                                  max_gap_s, min_peak_correlation,
                                  min_peak_margin,
                                  sample_hz_by_group=None,
                                  max_abs_drift_ppm=2_000.0,
                                  min_groups_per_window=1):
    """Run every usable sensor group around one coarse offset candidate."""
    all_groups = {**IMU_GROUPS, **ET_GROUPS}
    active = {
        name: features for name, features in all_groups.items()
        if all(feature in et_df.columns and feature in lsl_df.columns
               for feature in features)
    }
    if not active:
        return None

    et_start = float(et_df["timestamp"].min())
    et_end = float(et_df["timestamp"].max())
    lsl_seconds = lsl_df.index.view("int64") / 1e9 - float(prior_offset_s)
    start = max(et_start, float(np.min(lsl_seconds)))
    end = min(et_end, float(np.max(lsl_seconds)))
    if end <= start:
        return None

    if end - start <= window_s:
        centers = np.array([(start + end) / 2.0])
        effective_window_s = end - start
    else:
        centers = np.arange(start + window_s / 2.0,
                            end - window_s / 2.0 + step_s / 2.0,
                            step_s)
        effective_window_s = window_s

    raw_windows = []
    for center_s in centers:
        for group_name, features in active.items():
            result = _local_group_offset(
                et_df, lsl_df, features, group_name, center_s,
                prior_offset_s, effective_window_s, search_radius_s,
                max_gap_s, sample_hz_by_group)
            if result is not None:
                raw_windows.append(result)

    combined_windows = []
    for center_s in centers:
        group_results = [
            result for result in raw_windows
            if result["center_et_s"] == float(center_s)
            and result["peak_correlation"] >= min_peak_correlation
            and result["peak_margin"] >= min_peak_margin
            and not result.get("at_search_boundary", False)
        ]
        if not group_results:
            continue
        # Do not average two sensors that confidently support different lags.
        # Select the largest tight cluster; correlation breaks a one-vs-one tie.
        clusters = []
        for seed in group_results:
            members = [result for result in group_results
                       if abs(result["offset_s"] - seed["offset_s"])
                       <= LOCAL_GROUP_INLIER_THRESHOLD_S]
            offsets = np.asarray([result["offset_s"] for result in members])
            clusters.append((
                len(members),
                float(np.median([result["peak_correlation"] for result in members])),
                -float(np.std(offsets)),
                members,
            ))
        group_results = max(clusters, key=lambda cluster: cluster[:3])[3]
        if len(group_results) < int(min_groups_per_window):
            continue
        offsets = np.asarray([result["offset_s"] for result in group_results])
        combined_windows.append({
            "center_et_s": float(center_s),
            "offset_s": float(np.median(offsets)),
            "group_spread_s": float(np.std(offsets)),
            "peak_correlation": float(np.median(
                [result["peak_correlation"] for result in group_results])),
            "peak_margin": float(np.median(
                [result["peak_margin"] for result in group_results])),
            "groups": [result["group"] for result in group_results],
        })

    if not combined_windows:
        return {
            "prior_offset_s": float(prior_offset_s),
            "n_windows": int(len(centers)),
            "n_good_windows": 0,
            "good_window_ratio": 0.0,
            "temporal_coverage_ratio": 0.0,
            "windows": raw_windows,
            "group_windows": raw_windows,
        }

    fit = _fit_window_offsets(
        [window["center_et_s"] for window in combined_windows],
        [window["offset_s"] for window in combined_windows],
        max_abs_drift_ppm=max_abs_drift_ppm,
    )
    reference_et_s = float(np.median(
        [window["center_et_s"] for window in combined_windows]))
    groups_used = sorted({group for window in combined_windows
                          for group in window["groups"]})
    good_centers = np.asarray(
        [window["center_et_s"] for window in combined_windows], dtype=float)
    all_centers = np.asarray(centers, dtype=float)
    fit.update({
        "prior_offset_s": float(prior_offset_s),
        "reference_et_s": reference_et_s,
        "offset_at_reference_s": float(
            fit["intercept_s"] + (fit["scale"] - 1.0) * reference_et_s),
        "n_windows": int(len(centers)),
        "n_good_windows": int(len(combined_windows)),
        "good_window_ratio": float(len(combined_windows) / len(centers)),
        "temporal_coverage_ratio": float(
            np.ptp(good_centers) / np.ptp(all_centers)
            if len(good_centers) >= 2 and len(all_centers) >= 2
            and np.ptp(all_centers) > 0 else 0.0),
        "n_groups": int(len(groups_used)),
        "groups": groups_used,
        "median_peak_correlation": float(np.median(
            [window["peak_correlation"] for window in combined_windows])),
        "median_peak_margin": float(np.median(
            [window["peak_margin"] for window in combined_windows])),
        "median_group_spread_s": float(np.median(
            [window["group_spread_s"] for window in combined_windows])),
        "windows": combined_windows,
        "group_windows": raw_windows,
    })
    return fit


def estimate_windowed_alignment(et_df, lsl_df, initial_offset_s=None,
                                window_s=WINDOW_LENGTH_S,
                                step_s=WINDOW_STEP_S,
                                search_radius_s=LOCAL_SEARCH_RADIUS_S,
                                max_gap_s=MAX_INTERPOLATION_GAP_S,
                                min_peak_correlation=MIN_PEAK_CORRELATION,
                                min_peak_margin=MIN_PEAK_MARGIN,
                                verbose=False,
                                sample_hz_by_group=None,
                                max_abs_drift_ppm=2_000.0,
                                min_groups_per_window=1):
    # pylint: disable=too-many-arguments
    """Estimate an auditable affine ET-to-LSL mapping from local windows.

    A supplied trigger-derived offset is used only as a coarse search centre.
    Without it, the legacy full-recording candidates provide several coarse
    hypotheses and the one supported by the most good local windows wins.

    Long sampling gaps remain masked, each sensor uses an appropriate sample
    rate, correlation is overlap-normalised, and local offsets are robustly
    fitted for intercept plus clock drift.
    """
    if et_df is None or lsl_df is None or et_df.empty or lsl_df.empty:
        return None

    if initial_offset_s is not None and np.isfinite(initial_offset_s):
        priors = [float(initial_offset_s)]
    else:
        candidates = offset_candidates(et_df, lsl_df, verbose=False)
        priors = list(dict.fromkeys(float(candidate[0]) for candidate in candidates[:5]))
    if not priors:
        return None

    fits = [
        _windowed_alignment_for_prior(
            et_df, lsl_df, prior, window_s, step_s, search_radius_s,
            max_gap_s, min_peak_correlation, min_peak_margin,
            sample_hz_by_group, max_abs_drift_ppm,
            min_groups_per_window)
        for prior in priors
    ]
    fits = [fit for fit in fits if fit is not None]
    if not fits:
        return None

    def rank(fit):
        return (
            fit.get("n_good_windows", 0),
            fit.get("n_groups", 0),
            fit.get("median_peak_correlation", -np.inf),
            -fit.get("fit_residual_mad_s", np.inf),
        )

    best = max(fits, key=rank)
    if verbose:
        print(
            "windowed ET/LSL alignment: "
            f"{best.get('n_good_windows', 0)}/{best.get('n_windows', 0)} windows, "
            f"correlation={best.get('median_peak_correlation', np.nan):.3f}, "
            f"drift={best.get('drift_ppm', np.nan):.1f} ppm"
        )
    return best
