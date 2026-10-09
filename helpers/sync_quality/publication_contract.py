"""Lightweight, ordered schemas for compact synchronized publication files.

This module deliberately imports no modality implementation. In particular,
checking a DriverVideo header must not import OpenCV, MediaPipe, or model code.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence


DEVICE_TIMESTAMP_COLUMNS = {
    "Eyeball": (
        "eyeball_timestamp",
        "et_raw_timestamp",
        "lsl_timestamp",
    ),
    "DriverVideo": (
        "driver_video_timestamp",
        "et_raw_timestamp",
        "lsl_timestamp",
    ),
    "IDUN": (
        "idun_timestamp",
        "et_raw_timestamp",
        "lsl_timestamp",
    ),
}


# This is eye_detection.PUPIL_COLUMNS without its native Time_ms field. It is
# repeated here intentionally so importing the publication contract stays
# independent of OpenCV and all image-processing dependencies.
EYEBALL_SCIENCE_COLUMNS = (
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
)
EYEBALL_PUPIL_SOURCE_COLUMNS = ("Time_ms", *EYEBALL_SCIENCE_COLUMNS)


def _face_landmark_columns() -> tuple[str, ...]:
    return tuple(
        f"face_{index:03d}_{axis}"
        for index in range(478)
        for axis in ("x", "y", "z")
    )


def _pose_landmark_columns() -> tuple[str, ...]:
    return tuple(
        f"pose_{index:02d}_{field}"
        for index in range(33)
        for field in ("x", "y", "z", "visibility", "presence")
    )


def _pose_world_landmark_columns() -> tuple[str, ...]:
    return tuple(
        f"pose_world_{index:02d}_{axis}"
        for index in range(33)
        for axis in ("x", "y", "z")
    )


DRIVER_SCIENCE_COLUMNS = (
    "face_detected",
    "face_landmark_count",
    "pose_detected",
    "pose_landmark_count",
    *_face_landmark_columns(),
    *_pose_landmark_columns(),
    *_pose_world_landmark_columns(),
)


DEVICE_PUBLICATION_COLUMNS = {
    "Eyeball": (
        *DEVICE_TIMESTAMP_COLUMNS["Eyeball"],
        *EYEBALL_SCIENCE_COLUMNS,
    ),
    "DriverVideo": (
        "frame_index",
        *DEVICE_TIMESTAMP_COLUMNS["DriverVideo"],
        *DRIVER_SCIENCE_COLUMNS,
    ),
    "IDUN": (
        "sample_index",
        *DEVICE_TIMESTAMP_COLUMNS["IDUN"],
        "eeg_ch1",
    ),
}


def _device_value(mapping, device: str):
    try:
        return mapping[device]
    except KeyError as error:
        raise ValueError(
            f"Unknown publication device {device!r}; expected one of "
            f"{sorted(mapping)}"
        ) from error


def required_publication_columns(device: str) -> tuple[str, ...]:
    """Return the complete publication header in its required order."""

    return _device_value(DEVICE_PUBLICATION_COLUMNS, device)


def validate_exact_columns(
    columns: Sequence[str] | Iterable[str],
    expected: Sequence[str] | Iterable[str],
    *,
    label: str = "header",
) -> tuple[str, ...]:
    """Require exactly the expected columns, including order and uniqueness."""

    actual = tuple(str(column) for column in columns)
    required = tuple(str(column) for column in expected)
    if actual == required:
        return actual

    actual_counts = Counter(actual)
    required_set = set(required)
    actual_set = set(actual)
    missing = [column for column in required if column not in actual_set]
    extra = [column for column in actual if column not in required_set]
    duplicates = [
        column for column, count in actual_counts.items() if count > 1
    ]
    first_mismatch = next(
        (
            index
            for index, (actual_column, required_column) in enumerate(
                zip(actual, required)
            )
            if actual_column != required_column
        ),
        min(len(actual), len(required)),
    )
    details = [
        f"expected {len(required)} ordered columns, found {len(actual)}",
        f"first mismatch at position {first_mismatch}",
    ]
    if missing:
        details.append(f"missing={missing}")
    if extra:
        details.append(f"extra={extra}")
    if duplicates:
        details.append(f"duplicates={duplicates}")
    raise ValueError(
        f"{label} does not match the publication contract: "
        + "; ".join(details)
    )


__all__ = [
    "DEVICE_PUBLICATION_COLUMNS",
    "DEVICE_TIMESTAMP_COLUMNS",
    "DRIVER_SCIENCE_COLUMNS",
    "EYEBALL_PUPIL_SOURCE_COLUMNS",
    "EYEBALL_SCIENCE_COLUMNS",
    "required_publication_columns",
    "validate_exact_columns",
]
