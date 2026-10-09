"""Compose Driver-to-ET and ET-to-LSL affine mappings."""

from __future__ import annotations


import numpy as np
import pandas as pd


COMPOSED_MAPPING_COLUMNS = [
    "project", "pid", "visit", "segment_id", "mapping_status", "usable",
    "quality_grade", "review_required", "driver_to_lsl_intercept_s",
    "driver_to_lsl_scale", "et_lsl_bounding_et_start_s",
    "et_lsl_bounding_et_end_s", "driver_bounding_support_start_s",
    "driver_bounding_support_end_s", "driver_et_mapping_status",
    "et_lsl_mapping_status",
]

_GRADE_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}


def _one_row(mapping, label):
    if isinstance(mapping, pd.DataFrame):
        if len(mapping) != 1:
            raise ValueError(f"{label} must contain one row")
        return mapping.iloc[0]
    return mapping


def _text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _truthy(value) -> bool:
    return value if isinstance(value, bool) else _text(value).lower() in {
        "true", "1", "yes"
    }


def _number(row, name, label) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} has no numeric {name}") from error
    if not np.isfinite(value):
        raise ValueError(f"{label} has no numeric {name}")
    return value


def _optional_number(row, name, default) -> float:
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _worst_grade(first, second) -> str:
    grades = [_text(first).upper(), _text(second).upper()]
    grades = [grade for grade in grades if grade in _GRADE_RANK]
    return max(grades, key=_GRADE_RANK.__getitem__) if grades else ""


def _prepare_pair(driver_et_mapping, et_lsl_mapping):
    driver = _one_row(driver_et_mapping, "Driver/ET mapping")
    lsl = _one_row(et_lsl_mapping, "ET/LSL mapping")
    if not _truthy(driver.get("usable", False)):
        raise ValueError("Driver/ET mapping is not usable")
    if not _truthy(lsl.get("usable", False)):
        raise ValueError("ET/LSL mapping is not usable")

    driver_a = _number(driver, "driver_to_et_raw_intercept_s", "Driver/ET mapping")
    driver_b = _number(driver, "driver_to_et_raw_scale", "Driver/ET mapping")
    lsl_a = _number(lsl, "intercept_s", "ET/LSL mapping")
    lsl_b = _number(lsl, "scale", "ET/LSL mapping")
    if driver_b <= 0 or lsl_b <= 0:
        raise ValueError("mapping scale must be positive")

    et_start = max(
        _number(lsl, "et_start_s", "ET/LSL mapping"),
        (_number(lsl, "lsl_start_s", "ET/LSL mapping") - lsl_a) / lsl_b,
    )
    et_end = min(
        _number(lsl, "et_end_s", "ET/LSL mapping"),
        (_number(lsl, "lsl_end_s", "ET/LSL mapping") - lsl_a) / lsl_b,
    )
    driver_start = max(
        0.0,
        (et_start - driver_a) / driver_b,
        _optional_number(driver, "driver_interpolation_start_s", 0.0),
    )
    driver_end = min(
        (et_end - driver_a) / driver_b,
        _optional_number(driver, "driver_interpolation_end_s", np.inf),
    )
    # Bounds describe the observed overlap; an empty overlap does not prevent
    # composing accepted affine mappings for extrapolation.

    return driver, lsl, {
        "driver_a": driver_a,
        "driver_b": driver_b,
        "lsl_a": lsl_a,
        "lsl_b": lsl_b,
        "intercept": lsl_a + lsl_b * driver_a,
        "scale": lsl_b * driver_b,
        "et_start": et_start,
        "et_end": et_end,
        "driver_start": driver_start,
        "driver_end": driver_end,
    }


def compose_driver_lsl_mapping(driver_et_mapping, et_lsl_mapping) -> pd.Series:
    """Return the composed affine Driver-relative to LSL mapping."""
    driver, lsl, values = _prepare_pair(driver_et_mapping, et_lsl_mapping)
    row = {
        "project": _text(driver.get("project")),
        "pid": _text(driver.get("pid")),
        "visit": _text(driver.get("visit")),
        "segment_id": _text(driver.get("segment_id")),
        "mapping_status": "composed",
        "usable": True,
        "quality_grade": _worst_grade(
            driver.get("quality_grade"), lsl.get("quality_grade")
        ),
        "review_required": (
            _truthy(driver.get("review_required", False))
            or _truthy(lsl.get("review_required", False))
        ),
        "driver_to_lsl_intercept_s": values["intercept"],
        "driver_to_lsl_scale": values["scale"],
        "et_lsl_bounding_et_start_s": values["et_start"],
        "et_lsl_bounding_et_end_s": values["et_end"],
        "driver_bounding_support_start_s": values["driver_start"],
        "driver_bounding_support_end_s": values["driver_end"],
        "driver_et_mapping_status": driver.get("mapping_status", ""),
        "et_lsl_mapping_status": lsl.get("mapping_status", ""),
    }
    return pd.Series(row, index=COMPOSED_MAPPING_COLUMNS, dtype=object)


__all__ = [
    "COMPOSED_MAPPING_COLUMNS",
    "compose_driver_lsl_mapping",
]
