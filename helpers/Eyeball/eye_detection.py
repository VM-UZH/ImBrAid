"""Extract frame-level pupil measurements from four-panel Eyeball videos.

Dependencies: converted Eyeball MP4 files, OpenCV, NumPy and pandas."""

import os

import cv2
import numpy as np
import pandas as pd

VIDEO_SUBFOLDER = 'mp4_videos'
PUPIL_SIZE_SUBFOLDER = 'pupil_sizes'

PADDING_FRAMES = 7
EDGE_TO_CONNECT = 5

MAX_JUMP = 12

MIN_PUPIL_AREA = 20
MAX_PUPIL_AREA_RATIO = 0.35
MAX_AREA_JUMP_RATIO = 2.50
FILL_MAX_GAP = 5

MIN_CIRCULARITY_SOFT = 0.08
MIN_AXIS_RATIO_SOFT = 0.15
MIN_SOLIDITY_SOFT = 0.45

MAX_AREA_PENALTY_CAP = 1.5
MAX_DIST_PENALTY_CAP = 1.8

REACQUIRE_AFTER_LOW_FRAMES = 2

HIGH_SCORE_THRESHOLD = 0.85
MEDIUM_SCORE_THRESHOLD = 0.35

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

PUPIL_COLUMNS = [
    "Time_ms",
    "Left_Pupil_Area",
    "Right_Pupil_Area",
    "Left_Confidence",
    "Right_Confidence",
    *(f"{side}_{feature}" for side in ("Left", "Right")
      for feature in _EYE_FEATURE_NAMES),
]


def _empty_detection_features():
    return {feature_name: np.nan for feature_name in _EYE_FEATURE_NAMES}


def _get_area_from_roi(eye_roi, last_center=None, last_area=None):
    h_roi, w_roi = eye_roi.shape[:2]
    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

    gray_for_edges = cv2.addWeighted(gray, 1.5, gray, 0, -50)
    edges = cv2.Canny(gray_for_edges, 20, 60)

    y_indices = np.argmax(edges > 0, axis=0)
    x_indices = np.arange(w_roi)

    found_mask = y_indices > 0
    valid_x = x_indices[found_mask]
    valid_y = y_indices[found_mask]

    mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
    if len(valid_x) > 20:
        points = np.column_stack((valid_x, valid_y))
        top_corners = np.array([[w_roi - 1, 0], [0, 0]])
        poly_points = np.vstack([points, top_corners])
        cv2.fillPoly(mask, [poly_points.astype(np.int32)], 255)
    else:
        cv2.rectangle(mask, (0, 0), (w_roi, h_roi // 4), 255, -1)

    gray_masked = cv2.addWeighted(gray, 1.5, gray, 0, -10)
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
    if frame is None or frame.ndim < 2:
        raise ValueError("Cannot split eye ROIs from an empty frame")

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
    if not accepted:
        features = _empty_detection_features()
    return {
        f"{prefix}_{feature_name}": features[feature_name]
        for feature_name in _EYE_FEATURE_NAMES
    }


def _update_tracker_state(detected_area, detected_center, confidence,
                          last_center, last_area, low_streak):
    accepted = confidence in _ACCEPTED_CONFIDENCES
    if accepted:
        return detected_area, detected_center, detected_area, 0, True

    next_low_streak = low_streak + 1
    if next_low_streak >= REACQUIRE_AFTER_LOW_FRAMES:
        last_center, last_area = None, np.nan

    return np.nan, last_center, last_area, next_low_streak, False


def get_pupil_size(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot Open: {video_path}")
        cap.release()
        return pd.DataFrame()

    records = []
    try:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        sides = ("Left", "Right")
        trackers = {side: (None, np.nan, 0) for side in sides}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            rois = _split_eye_rois(frame, frame_width, frame_height)
            detections = [
                _get_area_from_roi(roi, last_center=trackers[side][0],
                                   last_area=trackers[side][1])
                for side, roi in zip(sides, rois)
            ]
            record = dict.fromkeys(PUPIL_COLUMNS)
            record["Time_ms"] = current_time_msec
            for side, detection in zip(sides, detections):
                detected_area, _, detected_center, confidence, features = detection
                area, center, last_area, low_streak, accepted = _update_tracker_state(
                    detected_area, detected_center, confidence, *trackers[side]
                )
                trackers[side] = (center, last_area, low_streak)
                record[f"{side}_Pupil_Area"] = area
                record[f"{side}_Confidence"] = confidence
                record.update(_prefixed_features(side, features, accepted))
            records.append(record)
    finally:
        cap.release()

    pupil_df = pd.DataFrame.from_records(records, columns=PUPIL_COLUMNS)

    for side in sides:
        _fill_short_missing(pupil_df, f"{side}_Pupil_Area", f"{side}_Confidence", FILL_MAX_GAP)

    return pupil_df


def process_video_folder(video_save_folder, overwrite=False, verbose=True,
                         video_subfolder=VIDEO_SUBFOLDER,
                         output_subfolder=PUPIL_SIZE_SUBFOLDER):
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
