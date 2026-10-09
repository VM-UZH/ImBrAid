"""Extract frame-level pupil measurements from four-panel Eyeball videos.

The first and last quarters of each decoded frame contain the left and right
eye.  ``get_pupil_size`` returns the accepted pupil areas, confidence labels,
and contour/ellipse features.  ``process_video_folder`` writes one CSV per MP4
and skips any existing output unless ``overwrite=True``.
"""

import os

import cv2
import numpy as np
import pandas as pd

# Sub-folders expected inside the study's Eyeball_Videos folder.
VIDEO_SUBFOLDER = 'mp4_videos'
PUPIL_SIZE_SUBFOLDER = 'pupil_sizes'

# Pixels cropped from every side of an eye panel, and the border width in
# which a contour still counts as "touching the edge" (such contours are
# eyelid/frame artefacts rather than pupils).
PADDING_FRAMES = 7
EDGE_TO_CONNECT = 5

# Expected per-frame movement of the pupil centre, in pixels.
MAX_JUMP = 12

MIN_PUPIL_AREA = 20
MAX_PUPIL_AREA_RATIO = 0.35
MAX_AREA_JUMP_RATIO = 2.50
FILL_MAX_GAP = 5

# Soft quality gates used for scoring (not hard rejection).
MIN_CIRCULARITY_SOFT = 0.08
MIN_AXIS_RATIO_SOFT = 0.15
MIN_SOLIDITY_SOFT = 0.45

# Caps for penalties so blink/saccade transitions are not over-penalized.
MAX_AREA_PENALTY_CAP = 1.5
MAX_DIST_PENALTY_CAP = 1.8

# Reacquire quickly after short tracking failures.
REACQUIRE_AFTER_LOW_FRAMES = 2

HIGH_SCORE_THRESHOLD = 0.85
MEDIUM_SCORE_THRESHOLD = 0.35

PUPIL_COLUMNS = [
    "Time_ms",
    "Left_Pupil_Area",
    "Right_Pupil_Area",
    "Left_Confidence",
    "Right_Confidence",
    "Left_Contour_Area",
    "Left_Ellipse_Area",
    "Left_Center_X",
    "Left_Center_Y",
    "Left_Major_Axis",
    "Left_Minor_Axis",
    "Left_Ellipse_Angle",
    "Left_Circularity",
    "Left_Solidity",
    "Left_Axis_Ratio",
    "Left_Quality_Score",
    "Right_Contour_Area",
    "Right_Ellipse_Area",
    "Right_Center_X",
    "Right_Center_Y",
    "Right_Major_Axis",
    "Right_Minor_Axis",
    "Right_Ellipse_Angle",
    "Right_Circularity",
    "Right_Solidity",
    "Right_Axis_Ratio",
    "Right_Quality_Score",
]

_ACCEPTED_CONFIDENCES = ("high", "medium")

_EYE_FEATURE_NAMES = (
    "Contour_Area",
    "Ellipse_Area",
    "Center_X",
    "Center_Y",
    "Major_Axis",
    "Minor_Axis",
    "Ellipse_Angle",
    "Circularity",
    "Solidity",
    "Axis_Ratio",
    "Quality_Score",
)


def _empty_detection_features():
    """Return a fresh, all-NaN feature record for one eye."""
    return {feature_name: np.nan for feature_name in _EYE_FEATURE_NAMES}


def _get_area_from_roi(eye_roi, last_center=None, last_area=None):
    """
    Find the pupil in a single eye panel and return its area.

    Args:
        eye_roi: BGR image of one (already padded/cropped) eye panel.
        last_center: (x, y) pupil centre of the previous frame, or None when
            the tracker has no history; used to penalize implausible jumps.
        last_area: pupil area of the previous frame, or NaN/None; used to
            penalize implausible area changes.

    Returns:
        (current_area, thresh, last_center, detection_confidence, features)
        where
        `current_area` is the winning candidate's contour area (NaN when no
        candidate passed the shape gates), `thresh` is the binary image the
        contours were taken from, `last_center` is the new centre (unchanged
        when nothing was found) and
        `detection_confidence` is "high", "medium" or "low". `features`
        contains the winning contour/ellipse measurements. Coordinates are
        relative to the padded single-eye ROI and all lengths/areas are in
        pixels/pixels squared.
    """
    h_roi, w_roi = eye_roi.shape[:2]
    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

    # ------ Remove the top black area to focus on the eye region ------
    gray_copy = gray.copy()
    low = 50
    gray_copy = cv2.addWeighted(gray_copy, 1.5, gray_copy, 0, -low)

    edges = cv2.Canny(gray_copy, 20, 60)

    # First edge pixel of every column: an approximation of the upper eyelid.
    y_indices = np.argmax(edges > 0, axis=0)
    x_indices = np.arange(w_roi)

    found_mask = y_indices > 0
    valid_x = x_indices[found_mask]
    valid_y = y_indices[found_mask]

    mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
    if len(valid_x) > 20:
        if len(valid_x) > 2:
            points = np.column_stack((valid_x, valid_y))
            top_corners = np.array([[w_roi - 1, 0], [0, 0]])
            poly_points = np.vstack([points, top_corners])
            cv2.fillPoly(mask, [poly_points.astype(np.int32)], 255)
    else:
        cv2.rectangle(mask, (0, 0), (w_roi, h_roi // 4), 255, -1)

    # ------ Enhance contrast and apply mask to keep black pupil region candidates -----
    gray_enhanced = cv2.addWeighted(gray, 1.5, gray, 0, -10)
    gray_masked = gray_enhanced.copy()
    gray_masked[mask == 255] = 255

    blurred = cv2.GaussianBlur(gray_masked, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_area = np.nan
    detection_confidence = "low"
    features = _empty_detection_features()
    best = None

    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        max_pupil_area = MAX_PUPIL_AREA_RATIO * (h_roi * w_roi)

        for contour in sorted_contours:
            area = cv2.contourArea(contour)
            if area < MIN_PUPIL_AREA:
                # Contours are sorted by area, so everything left is too small.
                break
            if area > max_pupil_area:
                continue

            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)
            touches_edge = (bbox_x <= EDGE_TO_CONNECT) or (bbox_y <= EDGE_TO_CONNECT) or \
                           (bbox_x + bbox_w >= w_roi - EDGE_TO_CONNECT) or \
                           (bbox_y + bbox_h >= h_roi - EDGE_TO_CONNECT)
            if touches_edge:
                continue

            if len(contour) < 5:
                # cv2.fitEllipse needs at least 5 points.
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            ellipse = cv2.fitEllipse(contour)
            (_, _), (axis_a, axis_b), ellipse_angle = ellipse
            if axis_a >= axis_b:
                major_axis = axis_a
                minor_axis = axis_b
                major_axis_angle = ellipse_angle % 180.0
            else:
                major_axis = axis_b
                minor_axis = axis_a
                major_axis_angle = (ellipse_angle + 90.0) % 180.0
            axis_ratio = (minor_axis / major_axis) if major_axis > 0 else 0
            ellipse_area = np.pi * (major_axis / 2.0) * (minor_axis / 2.0)

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = area / hull_area

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])

            dist = 0.0
            if last_center is not None:
                dist = np.sqrt((center_x - last_center[0]) ** 2 + (center_y - last_center[1]) ** 2)

            area_jump_ratio = 0.0
            if last_area is not None and not np.isnan(last_area) and last_area > 0:
                area_jump_ratio = abs(area - last_area) / last_area

            shape_ok = (
                circularity >= MIN_CIRCULARITY_SOFT
                and axis_ratio >= MIN_AXIS_RATIO_SOFT
                and solidity >= MIN_SOLIDITY_SOFT
            )
            if not shape_ok:
                continue

            if MAX_JUMP > 0:
                dist_penalty = min(dist / (MAX_JUMP * 2.0), MAX_DIST_PENALTY_CAP)
            else:
                dist_penalty = 0.0

            if MAX_AREA_JUMP_RATIO > 0:
                area_penalty = min(area_jump_ratio / MAX_AREA_JUMP_RATIO, MAX_AREA_PENALTY_CAP)
            else:
                area_penalty = 0.0

            shape_strength = circularity + axis_ratio + solidity
            score = shape_strength - 0.25 * dist_penalty - 0.10 * area_penalty
            if last_center is None:
                score += 0.15

            # Give a small bonus for strong shape even when the center moved quickly (saccade).
            if dist > MAX_JUMP * 2.0 and shape_strength > 1.20:
                score += 0.15

            candidate_features = {
                "Contour_Area": float(area),
                "Ellipse_Area": float(ellipse_area),
                "Center_X": float(center_x),
                "Center_Y": float(center_y),
                "Major_Axis": float(major_axis),
                "Minor_Axis": float(minor_axis),
                "Ellipse_Angle": float(major_axis_angle),
                "Circularity": float(circularity),
                "Solidity": float(solidity),
                "Axis_Ratio": float(axis_ratio),
                "Quality_Score": float(score),
            }

            if best is None or score > best[0]:
                best = (
                    score,
                    float(area),
                    (center_x, center_y),
                    candidate_features,
                )

        if best is not None:
            score, area, center, features = best
            current_area = area
            last_center = center
            if score >= HIGH_SCORE_THRESHOLD:
                detection_confidence = "high"
            elif score >= MEDIUM_SCORE_THRESHOLD:
                detection_confidence = "medium"
            else:
                detection_confidence = "low"

    return current_area, thresh, last_center, detection_confidence, features


def _split_eye_rois(frame, frame_width, frame_height):
    """
    Cut the padded left- and right-eye panels out of one four-panel frame.

    The left eye is the first quarter of the frame and the right eye is the
    last quarter; PADDING_FRAMES pixels are dropped on every side.
    """
    if frame is None or frame.ndim < 2:
        raise ValueError("Cannot split eye ROIs from an empty frame")

    # Trust the decoded frame over container metadata. Some codecs report a
    # zero or stale size even though decoding itself succeeds.
    actual_height, actual_width = frame.shape[:2]
    if frame_width != actual_width or frame_height != actual_height:
        frame_width = actual_width
        frame_height = actual_height

    left_panel_width = frame_width // 4
    if (
        frame_height <= 2 * PADDING_FRAMES
        or left_panel_width <= 2 * PADDING_FRAMES
    ):
        raise ValueError(
            f"Frame {frame_width}x{frame_height} is too small for "
            f"PADDING_FRAMES={PADDING_FRAMES}"
        )

    left_eye_roi = frame[
        PADDING_FRAMES:frame_height - PADDING_FRAMES,
        PADDING_FRAMES:left_panel_width - PADDING_FRAMES,
    ]
    right_eye_roi = frame[
        PADDING_FRAMES:frame_height - PADDING_FRAMES,
        3 * left_panel_width + PADDING_FRAMES:frame_width - PADDING_FRAMES,
    ]
    return left_eye_roi, right_eye_roi


def _fill_short_missing(pupil_df, area_col, conf_col, max_gap):
    """
    Interpolate runs of missing areas that are at most `max_gap` frames long.

    Filled samples are marked as "filled" in `conf_col`; longer gaps (real
    blinks or lasting tracking failures) are left as NaN. `pupil_df` is
    modified in place.
    """
    area_series = pupil_df[area_col].astype(float)
    missing = area_series.isna()
    groups = (missing != missing.shift(fill_value=False)).cumsum()
    run_length = missing.groupby(groups).transform("sum")
    short_missing = missing & (run_length <= max_gap)

    interpolated = area_series.interpolate(limit=max_gap, limit_direction='both')
    area_series.loc[short_missing] = interpolated.loc[short_missing]
    pupil_df[area_col] = area_series
    pupil_df.loc[short_missing, conf_col] = "filled"


def _prefixed_features(prefix, features, accepted):
    """Map one eye's features to CSV columns, blanking rejected candidates."""
    if not accepted:
        features = _empty_detection_features()
    return {
        f"{prefix}_{feature_name}": features[feature_name]
        for feature_name in _EYE_FEATURE_NAMES
    }


def _update_tracker_state(detected_area, detected_center, confidence,
                          last_center, last_area, low_streak):
    """Commit one detection to tracker history only when it is accepted.

    A rejected candidate must not become the temporal reference for the next
    frame.  After repeated low-confidence frames, the tracker drops its prior
    position and area so detection can reacquire without a temporal penalty.

    Returns:
        `(reported_area, next_center, next_area, next_low_streak, accepted)`.
        `reported_area` is NaN for rejected candidates, matching the CSV
        analysis signal.
    """
    accepted = confidence in _ACCEPTED_CONFIDENCES
    if accepted:
        return detected_area, detected_center, detected_area, 0, True

    next_low_streak = low_streak + 1
    next_center = last_center
    next_area = last_area
    if next_low_streak >= REACQUIRE_AFTER_LOW_FRAMES:
        next_center = None
        next_area = np.nan

    return np.nan, next_center, next_area, next_low_streak, False


def get_pupil_size(video_path):
    """
    Extract the per-frame pupil areas of both eyes from one eyeball video.

    Args:
        video_path: full path to the mp4 file.

    Returns:
        DataFrame with the backwards-compatible pupil-area/confidence columns
        plus the raw contour and fitted-ellipse features listed in
        `PUPIL_COLUMNS`. An empty DataFrame is returned when the video cannot
        be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot Open: {video_path}")
        cap.release()
        return pd.DataFrame()

    records = []
    try:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        left_last_center = None
        right_last_center = None
        left_last_area = np.nan
        right_last_area = np.nan
        left_low_streak = 0
        right_low_streak = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            left_eye_roi, right_eye_roi = _split_eye_rois(
                frame, frame_width, frame_height
            )

            (
                left_detected_area,
                _,
                left_detected_center,
                left_conf,
                left_features,
            ) = _get_area_from_roi(
                left_eye_roi,
                last_center=left_last_center,
                last_area=left_last_area,
            )
            (
                right_detected_area,
                _,
                right_detected_center,
                right_conf,
                right_features,
            ) = _get_area_from_roi(
                right_eye_roi,
                last_center=right_last_center,
                last_area=right_last_area,
            )

            (
                left_area,
                left_last_center,
                left_last_area,
                left_low_streak,
                left_accepted,
            ) = _update_tracker_state(
                left_detected_area,
                left_detected_center,
                left_conf,
                left_last_center,
                left_last_area,
                left_low_streak,
            )
            (
                right_area,
                right_last_center,
                right_last_area,
                right_low_streak,
                right_accepted,
            ) = _update_tracker_state(
                right_detected_area,
                right_detected_center,
                right_conf,
                right_last_center,
                right_last_area,
                right_low_streak,
            )

            record = {
                "Time_ms": current_time_msec,
                "Left_Pupil_Area": left_area,
                "Right_Pupil_Area": right_area,
                "Left_Confidence": left_conf,
                "Right_Confidence": right_conf,
            }
            record.update(_prefixed_features("Left", left_features, left_accepted))
            record.update(_prefixed_features("Right", right_features, right_accepted))
            records.append(record)
    finally:
        cap.release()

    pupil_df = pd.DataFrame.from_records(records, columns=PUPIL_COLUMNS)

    _fill_short_missing(pupil_df, "Left_Pupil_Area", "Left_Confidence", FILL_MAX_GAP)
    _fill_short_missing(pupil_df, "Right_Pupil_Area", "Right_Confidence", FILL_MAX_GAP)

    return pupil_df


def process_video_folder(video_save_folder, overwrite=False, verbose=True,
                         video_subfolder=VIDEO_SUBFOLDER,
                         output_subfolder=PUPIL_SIZE_SUBFOLDER):
    """
    Run `get_pupil_size` on every mp4 of a study and save one CSV per video.

    Reads `<video_save_folder>/mp4_videos/*.mp4` and writes
    `<video_save_folder>/pupil_sizes/<name>_pupil_sizes.csv`.

    Args:
        video_save_folder: the study's Eyeball_Videos folder.
        overwrite: re-process videos whose CSV already exists. By default any
            existing output path is skipped without opening or validating it.
        verbose: print the progress of every video.
        video_subfolder: where the mp4 files live inside `video_save_folder`.
        output_subfolder: where the CSV files are written inside
            `video_save_folder`. Pass '' for both to use the flat layout, in
            which the videos and their CSVs sit next to each other directly
            in `video_save_folder`.

    Returns:
        List of the CSV paths that were written during this call.
    """
    video_folder = os.path.join(video_save_folder, video_subfolder)
    output_folder = os.path.join(video_save_folder, output_subfolder)
    os.makedirs(output_folder, exist_ok=True)

    saved_paths = []
    for file_name in sorted(os.listdir(video_folder)):
        if not file_name.endswith('.mp4'):
            continue

        video_path = os.path.join(video_folder, file_name)
        csv_file_name = file_name.replace('.mp4', '_pupil_sizes.csv')
        save_path = os.path.join(output_folder, csv_file_name)
        if os.path.exists(save_path) and not overwrite:
            if verbose:
                print(f"CSV already exists for {file_name}, skipping...")
            continue

        if verbose:
            print(f"Processing {file_name}...")
        pupil_df = get_pupil_size(video_path)
        if not pupil_df.empty:
            pupil_df.to_csv(save_path, index=False)
            saved_paths.append(save_path)

    return saved_paths
