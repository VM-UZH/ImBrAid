import numpy as np
import pandas as pd

from helpers.project_utils.helper_utils import ScenarioIDs

# The driving modules that are analysed; the intro/outro scenarios are not.
SCENARIO_IDS = list(ScenarioIDs)

# Blink detection, as fractions of the rolling-median baseline.
CLOSE_THRESHOLD = 0.25
OPEN_THRESHOLD = 0.40
BASELINE_WINDOW = 30

MERGE_GAP_S = 0.06
MIN_EVENT_S = 0.05
MAX_EVENT_S = 0.80

# Both eyes have to close for a blink to count, but they are detected a frame
# or two apart: the two panels differ in lighting, so each pupil is lost at a
# slightly different frame. Requiring the same frame for both drops most real
# blinks, so each eye's closure is widened by this many frames before the two
# are combined. Anything from 2 to 5 gives practically the same result.
EYE_SYNC_TOLERANCE_FRAMES = 3

# Whether low-confidence pupil samples may still take part in blink detection.
# A blink is exactly where the detector loses the pupil, so excluding them
# would throw away the samples the blink is made of.
INCLUDE_LOW_CONFIDENCE = True
VALID_CONFIDENCE_STRICT = {'high', 'medium', 'filled'}
VALID_CONFIDENCE_ALL = {'high', 'medium', 'filled', 'low'}

# Below this a scenario is treated as interrupted and its blink numbers dropped.
MIN_SCENARIO_DURATION_S = 240


def _widen(mask, frames):
    """
    Widen a boolean mask by `frames` samples in both directions.

    Used so that "both eyes closed" means the two closures overlap within a
    few frames, rather than falling on the very same sample.
    """
    if frames <= 0:
        return mask
    width = 2 * frames + 1
    widened = mask.astype(float).rolling(width, center=True, min_periods=1).max()
    return widened.fillna(0) > 0


def detect_blink_events(pupil_df, close_th=CLOSE_THRESHOLD, open_th=OPEN_THRESHOLD,
                        baseline_win=BASELINE_WINDOW, merge_gap=MERGE_GAP_S,
                        min_dur=MIN_EVENT_S, max_dur=MAX_EVENT_S,
                        sync_tolerance=EYE_SYNC_TOLERANCE_FRAMES):
    """
    Find the blinks in one visit's pupil trace.

    Both eyes have to close and re-open together, which rejects the
    single-eye dropouts the detector produces when one pupil is briefly lost.
    "Together" allows `sync_tolerance` frames of slack, because the two eyes
    are rarely lost on the identical frame; requiring that drops most real
    blinks and can leave a whole scenario at zero.

    Returns:
        (pupil_df sorted by time, list of (start, end, duration_s) events,
        boolean mask of the samples inside a kept event).
    """
    pupil_df = pupil_df.sort_values('real_time').reset_index(drop=True).copy()

    if 'Left_Confidence' in pupil_df.columns and 'Right_Confidence' in pupil_df.columns:
        labels = VALID_CONFIDENCE_ALL if INCLUDE_LOW_CONFIDENCE else VALID_CONFIDENCE_STRICT
        left_ok = pupil_df['Left_Confidence'].astype(str).str.lower().isin(labels)
        right_ok = pupil_df['Right_Confidence'].astype(str).str.lower().isin(labels)
        valid_signal = left_ok & right_ok
    else:
        # Files written before the confidence columns existed.
        valid_signal = pd.Series(True, index=pupil_df.index)

    left = pupil_df['Left_Pupil_Area'].where(
        (pupil_df['Left_Pupil_Area'] > 0) & valid_signal, np.nan)
    right = pupil_df['Right_Pupil_Area'].where(
        (pupil_df['Right_Pupil_Area'] > 0) & valid_signal, np.nan)

    left_ratio = left / left.rolling(window=baseline_win, min_periods=10).median().ffill().bfill()
    right_ratio = right / right.rolling(window=baseline_win, min_periods=10).median().ffill().bfill()

    # A lost pupil is what a blink looks like: NaN areas count as closed, not as
    # missing evidence. Long tracking losses are dropped later by MAX_EVENT_S.
    left_missing = pupil_df['Left_Pupil_Area'].isna()
    right_missing = pupil_df['Right_Pupil_Area'].isna()

    left_raw = (((left_ratio < close_th) | left_missing) & valid_signal).fillna(False)
    right_raw = (((right_ratio < close_th) | right_missing) & valid_signal).fillna(False)

    # Widened masks decide *whether* both eyes took part in a blink; the raw
    # masks decide *how long* it lasted, so the tolerance cannot stretch the
    # measured duration.
    closed = (_widen(left_raw, sync_tolerance) & _widen(right_raw, sync_tolerance)).to_numpy()
    either_closed = (left_raw | right_raw).to_numpy()

    opened = ((left_ratio > open_th) & (right_ratio > open_th)
              & valid_signal).fillna(False).to_numpy()
    times = pupil_df['real_time'].to_numpy()

    # Events are collected as index ranges, so that merging and trimming can
    # both work on them before any duration is measured.
    raw_events = []
    start_idx = None
    for i in range(len(pupil_df)):
        if start_idx is None and closed[i]:
            start_idx = i
        elif start_idx is not None and opened[i]:
            raw_events.append([start_idx, i])
            start_idx = None
    if start_idx is not None:
        raw_events.append([start_idx, len(pupil_df) - 1])

    merged_events = []
    for from_idx, to_idx in raw_events:
        if merged_events and (pd.Timestamp(times[from_idx])
                              - pd.Timestamp(times[merged_events[-1][1]])
                              ).total_seconds() <= merge_gap:
            merged_events[-1][1] = max(to_idx, merged_events[-1][1])
        else:
            merged_events.append([from_idx, to_idx])

    valid_events = []
    blink_mask = np.zeros(len(pupil_df), dtype=bool)
    for from_idx, to_idx in merged_events:
        # Shrink to where an eye was actually below the threshold, so the
        # tolerance can never stretch the measured duration.
        actual = np.flatnonzero(either_closed[from_idx:to_idx + 1])
        if len(actual) == 0:
            continue
        from_idx, to_idx = from_idx + actual[0], from_idx + actual[-1]

        start_t, end_t = pd.Timestamp(times[from_idx]), pd.Timestamp(times[to_idx])
        duration_s = (end_t - start_t).total_seconds()
        if min_dur <= duration_s <= max_dur:
            valid_events.append((start_t, end_t, duration_s))
            blink_mask[from_idx:to_idx + 1] = True

    return pupil_df, valid_events, blink_mask


def summarize_blinks_by_scenario(pupil_df, min_dur=MIN_EVENT_S, max_dur=MAX_EVENT_S):
    """
    Count and time the blinks of one visit, per scenario.

    Returns:
        DataFrame with ScenarioID, Blink_Total_Duration_s, Blink_Count,
        Blink_Average_Length_s and Scenario_Duration_s.
    """
    scenario_rows = []
    for scenario_id, group in pupil_df.groupby('ScenarioID', dropna=True):
        group = group.sort_values('real_time').reset_index(drop=True)
        if group.empty:
            continue

        if len(group) > 1:
            scenario_duration_s = (group['real_time'].iloc[-1]
                                   - group['real_time'].iloc[0]).total_seconds()
        else:
            scenario_duration_s = np.nan

        is_blink = group['blink'].fillna(False).astype(bool).to_numpy()
        starts = np.where(is_blink & ~np.r_[False, is_blink[:-1]])[0]
        ends = np.where(is_blink & ~np.r_[is_blink[1:], False])[0]

        durations = []
        for start_idx, end_idx in zip(starts, ends):
            duration_s = (group['real_time'].iloc[end_idx]
                          - group['real_time'].iloc[start_idx]).total_seconds()
            if min_dur <= duration_s <= max_dur:
                durations.append(duration_s)

        total_s = float(np.sum(durations)) if durations else 0.0
        scenario_rows.append({
            'ScenarioID': int(scenario_id),
            'Blink_Total_Duration_s': total_s,
            'Blink_Count': len(durations),
            'Blink_Average_Length_s': (total_s / len(durations)) if durations else np.nan,
            'Scenario_Duration_s': scenario_duration_s,
        })

    return pd.DataFrame(scenario_rows)


