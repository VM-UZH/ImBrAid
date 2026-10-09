"""Add Eyeball, ET-raw and LSL timestamps to pupil-size CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .publication_contract import (
    EYEBALL_PUPIL_SOURCE_COLUMNS,
    validate_exact_columns,
)

SUMMARY_COLUMNS = [
    "project", "pid", "visit", "processing_status", "reason",
    "source_pupil_csv", "output_pupil_csv", "rows", "aligned_rows",
    "aligned_fraction", "supported_rows", "extrapolated_rows",
    "eyeball_start_s", "eyeball_end_s",
    "common_et_support_start_s", "common_et_support_end_s",
    "et_raw_start_s", "et_raw_end_s", "lsl_start_s", "lsl_end_s",
    "quality_grade", "review_required", "et_eyeball_mapping_status",
    "et_lsl_mapping_status",
]

_GRADE_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}


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


def _worst_grade(first, second) -> str:
    grades = [_text(first).upper(), _text(second).upper()]
    grades = [grade for grade in grades if grade in _GRADE_RANK]
    return max(grades, key=_GRADE_RANK.__getitem__) if grades else ""


def _prepare_pair(et_eyeball_mapping, et_lsl_mapping):
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

    support_start = max(
        _number(eye, "et_start_s", "ET/Eyeball mapping"),
        _number(lsl, "et_start_s", "ET/LSL mapping"),
        (_number(lsl, "lsl_start_s", "ET/LSL mapping") - lsl_a) / lsl_b,
    )
    support_end = min(
        _number(eye, "et_end_s", "ET/Eyeball mapping"),
        _number(lsl, "et_end_s", "ET/LSL mapping"),
        (_number(lsl, "lsl_end_s", "ET/LSL mapping") - lsl_a) / lsl_b,
    )
    pupil_path = Path(_text(eye.get("pupil_csv_path"))).expanduser().resolve()
    return eye, lsl, {
        "project": _text(eye.get("project")),
        "pid": _text(eye.get("pid")),
        "visit": _text(eye.get("visit")),
        "pupil_path": pupil_path,
        "eye_a": eye_a,
        "eye_b": eye_b,
        "lsl_a": lsl_a,
        "lsl_b": lsl_b,
        "support_start": support_start,
        "support_end": support_end,
        "quality_grade": _worst_grade(
            eye.get("quality_grade"), lsl.get("quality_grade")
        ),
        "review_required": (
            _truthy(eye.get("review_required", False))
            or _truthy(lsl.get("review_required", False))
        ),
    }


def build_eyeball_timestamp_table(
    et_eyeball_mapping,
    et_lsl_mapping,
):
    """Apply accepted clock mappings to all finite pupil times, extrapolating."""
    eye_row, lsl_row, meta = _prepare_pair(
        et_eyeball_mapping, et_lsl_mapping
    )
    pupil = pd.read_csv(meta["pupil_path"], low_memory=False)
    validate_exact_columns(
        pupil.columns,
        EYEBALL_PUPIL_SOURCE_COLUMNS,
        label="Eyeball pupil source header",
    )

    eye_seconds = pd.to_numeric(
        pupil.pop("Time_ms"), errors="coerce"
    ).to_numpy(float) / 1000.0
    et_seconds = (eye_seconds - meta["eye_a"]) / meta["eye_b"]
    lsl_seconds = meta["lsl_a"] + meta["lsl_b"] * et_seconds
    mapped = np.isfinite(eye_seconds) & np.isfinite(et_seconds) & np.isfinite(lsl_seconds)
    in_support = (
        mapped & (et_seconds >= meta["support_start"])
        & (et_seconds <= meta["support_end"])
    )
    # Support is reported for QC; it does not limit the exported timestamps.
    et_output = np.where(mapped, et_seconds, np.nan)
    lsl_output = np.where(mapped, lsl_seconds, np.nan)

    output = pupil.copy()
    output.insert(0, "lsl_timestamp", lsl_output)
    output.insert(0, "et_raw_timestamp", et_output)
    output.insert(0, "eyeball_timestamp", eye_seconds)

    meta.update({
        "rows": len(output),
        "aligned_rows": int(mapped.sum()),
        "aligned_fraction": float(mapped.mean()) if len(output) else 0.0,
        "supported_rows": int(in_support.sum()),
        "extrapolated_rows": int((mapped & ~in_support).sum()),
        "eyeball_start_s": float(eye_seconds[0]) if len(output) else np.nan,
        "eyeball_end_s": float(eye_seconds[-1]) if len(output) else np.nan,
        "et_raw_start_s": float(np.nanmin(et_output)) if mapped.any() else np.nan,
        "et_raw_end_s": float(np.nanmax(et_output)) if mapped.any() else np.nan,
        "lsl_start_s": float(np.nanmin(lsl_output)) if mapped.any() else np.nan,
        "lsl_end_s": float(np.nanmax(lsl_output)) if mapped.any() else np.nan,
        "eye_row": eye_row,
        "lsl_row": lsl_row,
    })
    return output, meta


def _summary_row(eye_row, lsl_row, *, status, reason="", output_path="", meta=None):
    meta = meta or {}
    return {
        "project": _text(eye_row.get("project")),
        "pid": _text(eye_row.get("pid")),
        "visit": _text(eye_row.get("visit")),
        "processing_status": status,
        "reason": reason,
        "source_pupil_csv": _text(eye_row.get("pupil_csv_path")),
        "output_pupil_csv": str(output_path),
        "rows": meta.get("rows", np.nan),
        "aligned_rows": meta.get("aligned_rows", np.nan),
        "aligned_fraction": meta.get("aligned_fraction", np.nan),
        "supported_rows": meta.get("supported_rows", np.nan),
        "extrapolated_rows": meta.get("extrapolated_rows", np.nan),
        "eyeball_start_s": meta.get("eyeball_start_s", np.nan),
        "eyeball_end_s": meta.get("eyeball_end_s", np.nan),
        "common_et_support_start_s": meta.get("support_start", np.nan),
        "common_et_support_end_s": meta.get("support_end", np.nan),
        "et_raw_start_s": meta.get("et_raw_start_s", np.nan),
        "et_raw_end_s": meta.get("et_raw_end_s", np.nan),
        "lsl_start_s": meta.get("lsl_start_s", np.nan),
        "lsl_end_s": meta.get("lsl_end_s", np.nan),
        "quality_grade": meta.get("quality_grade", np.nan),
        "review_required": meta.get("review_required", np.nan),
        "et_eyeball_mapping_status": eye_row.get("mapping_status", np.nan),
        "et_lsl_mapping_status": lsl_row.get("mapping_status", np.nan),
    }


def export_aligned_eyeball_csvs(
    et_eyeball_alignments,
    et_lsl_alignments,
    *,
    output_dir,
    overwrite=False,
):
    """Write one aligned pupil CSV for every available mapping pair."""
    output_dir = Path(output_dir)
    lsl_by_key = {
        tuple(_text(row[column]) for column in ("project", "pid", "visit")): row
        for _, row in et_lsl_alignments.iterrows()
    }
    rows = []
    for _, eye_row in et_eyeball_alignments.iterrows():
        key = tuple(_text(eye_row[column]) for column in ("project", "pid", "visit"))
        lsl_row = lsl_by_key.get(key)
        if lsl_row is None:
            rows.append(_summary_row(
                eye_row, {}, status="unavailable",
                reason="no matching ET/LSL mapping row",
            ))
            continue
        try:
            table, meta = build_eyeball_timestamp_table(eye_row, lsl_row)
            name = f"{meta['pid']}_{meta['visit']}_eyeball_synchronized.csv"
            output_path = output_dir / name
            if output_path.exists() and not overwrite:
                status = "skipped_existing"
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                table.to_csv(output_path, index=False)
                status = "written"
            rows.append(_summary_row(
                eye_row, lsl_row, status=status,
                output_path=output_path, meta=meta,
            ))
        except Exception as error:
            rows.append(_summary_row(
                eye_row, lsl_row, status="unavailable",
                reason=f"{type(error).__name__}: {error}",
            ))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def save_eyeball_timestamp_summary(summary, path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).reindex(columns=SUMMARY_COLUMNS).to_csv(path, index=False)
    return path


__all__ = [
    "SUMMARY_COLUMNS",
    "build_eyeball_timestamp_table",
    "export_aligned_eyeball_csvs",
    "save_eyeball_timestamp_summary",
]
