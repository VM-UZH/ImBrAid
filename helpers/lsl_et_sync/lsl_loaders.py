import numpy as np
import pandas as pd
import pyxdf

from helpers import helper_lsl as hl


def load_lsl_xdf(xdf_path):
    """Load the native-rate ET stream from one explicitly selected XDF."""

    lsl_data, _ = pyxdf.load_xdf(xdf_path)
    lsl_df = _available_et_data(lsl_data)
    if lsl_df is None:
        return None, None

    return lsl_df, lsl_data


def _available_et_data(lsl_data):
    """Return every native ET group present in an XDF, without upsampling."""
    frames = []
    for feature in hl.ET_FEATURES:
        stream = hl.find_stream(lsl_data, feature)
        if stream is None:
            continue
        values = stream["time_series"]
        timestamps = pd.to_datetime(stream["time_stamps"], unit="s")
        header = stream["info"]["desc"][0]["channels"][0]["channel"]
        names = [channel["label"][0] for channel in header]
        frame = pd.DataFrame(values, columns=names, index=timestamps)
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame[frame > 1e9] = float("nan")
        frames.append(frame)
    return pd.concat(frames, axis=1).sort_index() if frames else None


def _index_seconds(index):
    """Return finite index values as seconds without assuming one index type."""
    if isinstance(index, pd.DatetimeIndex):
        values = index.asi8.astype(np.float64) / 1e9
    else:
        values = pd.to_numeric(pd.Index(index), errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _stream_time_bounds(stream):
    """Return one raw XDF stream's finite start/end timestamps, if available."""
    timestamps = np.asarray(stream.get("time_stamps", []), dtype=float)
    timestamps = timestamps[np.isfinite(timestamps)]
    if timestamps.size == 0:
        return None
    return float(np.min(timestamps)), float(np.max(timestamps))


def et_stream_coverage(lsl_df, lsl_data, *, same_clock_tolerance_s=None):
    """
    Fraction of an LSL recording that the ET stream actually covers.

    Numerator: the span of the ET stream inside the file. Denominator: the
    recording span formed by raw XDF streams in the same clock domain as the
    ET stream. Streams whose intervals are far from the ET interval are
    excluded; this prevents Unix-epoch devices such as IDUN from being mixed
    with local-LSL-clock streams. A low value means the glasses stopped early
    or dropped out while the rest of the same-clock recording continued.

    Args:
        lsl_df: ET stream DataFrame from `load_lsl_xdf`.
        lsl_data: the raw xdf stream list of the same recording.
        same_clock_tolerance_s: maximum permitted separation between a raw
            stream interval and the ET interval. By default this is one ET
            span, bounded to 60--3600 seconds. Overlapping intervals always
            qualify.

    Returns:
        Coverage between 0 and 1. Returns 0 when the ET interval is empty or
        has no positive duration.
    """
    if lsl_df is None or lsl_df.empty:
        return 0.0

    et_seconds = _index_seconds(lsl_df.index)
    if et_seconds.size == 0:
        return 0.0
    et_start = float(np.min(et_seconds))
    et_end = float(np.max(et_seconds))
    et_span = et_end - et_start
    if not np.isfinite(et_span) or et_span <= 0:
        return 0.0

    if same_clock_tolerance_s is None:
        tolerance = max(60.0, min(3600.0, et_span))
    else:
        tolerance = float(same_clock_tolerance_s)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("same_clock_tolerance_s must be finite and non-negative")

    # Seed the bounds with the ET interval itself. Besides making the metric
    # well-defined when the XDF stream list is sparse, this guarantees that
    # floating-point or clock-domain anomalies cannot produce coverage > 1.
    starts = [et_start]
    ends = [et_end]
    for stream in lsl_data or []:
        bounds = _stream_time_bounds(stream)
        if bounds is None:
            continue
        stream_start, stream_end = bounds
        separated_by = max(et_start - stream_end, stream_start - et_end, 0.0)
        if separated_by <= tolerance:
            starts.append(stream_start)
            ends.append(stream_end)

    recording_span = max(ends) - min(starts)
    if not np.isfinite(recording_span) or recording_span <= 0:
        return 0.0
    return float(et_span / recording_span)
