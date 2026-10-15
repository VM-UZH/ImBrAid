"""Load the ET stream from XDF and measure its recording coverage.

Dependencies: XDF files, pyxdf, NumPy and pandas."""

import numpy as np
import pandas as pd
import pyxdf

from helpers import helper_lsl as hl


def load_lsl_xdf(xdf_path):
    lsl_data, _ = pyxdf.load_xdf(xdf_path)
    lsl_df = _available_et_data(lsl_data)
    if lsl_df is None:
        return None, None

    return lsl_df, lsl_data


def _available_et_data(lsl_data):
    frames = []
    for feature in hl.ET_FEATURES:
        stream = hl.find_stream(lsl_data, feature)
        if stream is None:
            continue
        timestamps = pd.to_datetime(stream["time_stamps"], unit="s")
        header = stream["info"]["desc"][0]["channels"][0]["channel"]
        names = [channel["label"][0] for channel in header]
        values = stream["time_series"] if len(timestamps) else np.empty((0, len(names)))
        frame = pd.DataFrame(values, columns=names, index=timestamps)
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame[frame > 1e9] = float("nan")
        frames.append(frame)
    return pd.concat(frames, axis=1).sort_index() if frames else None


def _index_seconds(index):
    if isinstance(index, pd.DatetimeIndex):
        values = index.asi8.astype(np.float64) / 1e9
    else:
        values = pd.to_numeric(pd.Index(index), errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _stream_time_bounds(stream):
    timestamps = np.asarray(stream.get("time_stamps", []), dtype=float)
    timestamps = timestamps[np.isfinite(timestamps)]
    if timestamps.size == 0:
        return None
    return float(np.min(timestamps)), float(np.max(timestamps))


def et_stream_coverage(lsl_df, lsl_data, *, same_clock_tolerance_s=None):
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
