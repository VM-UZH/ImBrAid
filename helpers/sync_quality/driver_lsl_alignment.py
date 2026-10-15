"""Compose accepted Driver-to-ET and ET-to-LSL clock mappings.

Dependencies: Driver-to-ET and ET-to-LSL alignment tables, NumPy and
pandas."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _one_row(mapping, label):
    if isinstance(mapping, pd.DataFrame):
        if len(mapping) != 1:
            raise ValueError(f"{label} must contain one row")
        return mapping.iloc[0]
    return mapping


def _truthy(value):
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _number(row, name, label):
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} has no numeric {name}") from error
    if not np.isfinite(value):
        raise ValueError(f"{label} has no numeric {name}")
    return value


def compose_driver_lsl_mapping(driver_et_mapping, et_lsl_mapping) -> pd.Series:
    driver = _one_row(driver_et_mapping, "Driver/ET mapping")
    lsl = _one_row(et_lsl_mapping, "ET/LSL mapping")
    if not _truthy(driver.get("usable", False)):
        raise ValueError("Driver/ET mapping is not usable")
    if not _truthy(lsl.get("usable", False)):
        raise ValueError("ET/LSL mapping is not usable")
    da = _number(driver, "driver_to_et_raw_intercept_s", "Driver/ET mapping")
    db = _number(driver, "driver_to_et_raw_scale", "Driver/ET mapping")
    la = _number(lsl, "intercept_s", "ET/LSL mapping")
    lb = _number(lsl, "scale", "ET/LSL mapping")
    if db <= 0 or lb <= 0:
        raise ValueError("mapping scale must be positive")
    for name in ("et_start_s", "et_end_s", "lsl_start_s", "lsl_end_s"):
        _number(lsl, name, "ET/LSL mapping")
    return pd.Series(
        {
            "driver_to_lsl_intercept_s": la + lb * da,
            "driver_to_lsl_scale": lb * db,
        }
    )


__all__ = ["compose_driver_lsl_mapping"]
