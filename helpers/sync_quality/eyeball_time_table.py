"""Export pupil samples with ET and LSL timestamps.

Dependencies: pupil CSVs and 01/03 alignment tables, NumPy and pandas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .publication_contract import (
    EYEBALL_PUPIL_SOURCE_COLUMNS,
    validate_exact_columns,
)


def _text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _truthy(value) -> bool:
    return value if isinstance(value, bool) else _text(value).lower() in {
        "true", "1", "yes"
    }


def _one_row(mapping, label):
    if isinstance(mapping, pd.DataFrame):
        if len(mapping) != 1:
            raise ValueError(f"{label} must contain one row")
        return mapping.iloc[0]
    return mapping


def _number(row, name, label) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} has no numeric {name}") from error
    if not np.isfinite(value):
        raise ValueError(f"{label} has no numeric {name}")
    return value


def build_eyeball_timestamp_table(et_eyeball_mapping, et_lsl_mapping):
    eye = _one_row(et_eyeball_mapping, "ET/Eyeball mapping")
    lsl = _one_row(et_lsl_mapping, "ET/LSL mapping")
    if not _truthy(eye.get("usable", False)):
        raise ValueError("ET/Eyeball mapping is not usable")
    if not _truthy(lsl.get("usable", False)):
        raise ValueError("ET/LSL mapping is not usable")
    eye_a = _number(eye, "intercept_s", "ET/Eyeball mapping")
    eye_b = _number(eye, "scale", "ET/Eyeball mapping")
    lsl_a = _number(lsl, "intercept_s", "ET/LSL mapping")
    lsl_b = _number(lsl, "scale", "ET/LSL mapping")
    if eye_b <= 0 or lsl_b <= 0:
        raise ValueError("mapping scale must be positive")
    for row, name, label in (
        (eye, "et_start_s", "ET/Eyeball mapping"),
        (lsl, "et_start_s", "ET/LSL mapping"),
        (lsl, "lsl_start_s", "ET/LSL mapping"),
        (eye, "et_end_s", "ET/Eyeball mapping"),
        (lsl, "et_end_s", "ET/LSL mapping"),
        (lsl, "lsl_end_s", "ET/LSL mapping"),
    ):
        _number(row, name, label)
    pupil_path = Path(_text(eye.get("pupil_csv_path"))).expanduser().resolve()
    pupil = pd.read_csv(pupil_path, low_memory=False)
    validate_exact_columns(
        pupil.columns, EYEBALL_PUPIL_SOURCE_COLUMNS,
        label="Eyeball pupil source header",
    )
    eye_seconds = pd.to_numeric(
        pupil.pop("Time_ms"), errors="coerce"
    ).to_numpy(float) / 1000.0
    et_seconds = (eye_seconds - eye_a) / eye_b
    lsl_seconds = lsl_a + lsl_b * et_seconds
    mapped = np.isfinite(eye_seconds) & np.isfinite(et_seconds) & np.isfinite(lsl_seconds)
    output = pupil.copy()
    output.insert(0, "lsl_timestamp", np.where(mapped, lsl_seconds, np.nan))
    output.insert(0, "et_raw_timestamp", np.where(mapped, et_seconds, np.nan))
    output.insert(0, "eyeball_timestamp", eye_seconds)
    return output


def export_aligned_eyeball_csvs(
    et_eyeball_alignments, et_lsl_alignments, *, output_dir, overwrite=False,
):
    output_dir = Path(output_dir)
    lsl_by_key = {
        tuple(_text(row[column]) for column in ("project", "pid", "visit")): row
        for _, row in et_lsl_alignments.iterrows()
    }
    for _, eye_row in et_eyeball_alignments.iterrows():
        key = tuple(_text(eye_row[column]) for column in ("project", "pid", "visit"))
        label = f"{key[1]}/{key[2]}"
        lsl_row = lsl_by_key.get(key)
        if lsl_row is None:
            print(f"{label}: unavailable (no matching ET/LSL mapping row)")
            continue
        try:
            table = build_eyeball_timestamp_table(eye_row, lsl_row)
            output_path = output_dir / f"{key[1]}_{key[2]}_eyeball_synchronized.csv"
            if output_path.exists() and not overwrite:
                print(f"{label}: skipped_existing ({output_path})")
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                table.to_csv(output_path, index=False)
                print(f"{label}: written ({output_path})")
        except Exception as error:
            print(f"{label}: unavailable ({type(error).__name__}: {error})")


__all__ = ["build_eyeball_timestamp_table", "export_aligned_eyeball_csvs"]
