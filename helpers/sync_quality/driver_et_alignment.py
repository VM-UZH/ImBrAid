"""Build Driver-to-ET clock mappings from saved panel-transition
annotations.

Dependencies: panel annotation tables, NumPy and pandas."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import panel_annotation as panel


GROUP_COLUMNS = ["project", "pid", "visit", "segment_id"]
EVENTS = ("panel_off", "panel_on")

ALIGNMENT_COLUMNS = [
    "project",
    "pid",
    "visit",
    "segment_id",
    "mapping_status",
    "usable",
    "quality_grade",
    "review_required",
    "mapping_reason",
    "anchor_count",
    "driver_to_et_raw_intercept_s",
    "driver_to_et_raw_scale",
    "driver_to_et_raw_drift_ppm",
    "driver_interpolation_start_s",
    "driver_interpolation_end_s",
    "warnings",
]


def _missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _text(value: object) -> str:
    return "" if _missing(value) else str(value).strip()


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _base_row(key: Sequence[str]) -> dict[str, object]:
    row = {column: np.nan for column in ALIGNMENT_COLUMNS}
    row.update(
        {
            **dict(zip(GROUP_COLUMNS, key)),
            "mapping_status": "unavailable",
            "usable": False,
            "quality_grade": "D",
            "review_required": False,
            "mapping_reason": "no paired accepted panel anchors",
            "anchor_count": 0,
            "warnings": "",
        }
    )
    return row


def _group_key(values: Sequence[object]) -> tuple[str, str, str, str]:
    if len(values) != 4:
        raise ValueError("alignment group key must have four values")
    result = tuple(_text(value) for value in values)
    if any(not value for value in result):
        raise ValueError(f"empty alignment group key: {result}")
    return result  # type: ignore[return-value]


def _anchor_from_row(
    annotation: pd.Series, *, source: str
) -> dict[str, object]:
    relative, raw, uncertainty, _ = panel.annotation_bracket_values(annotation)
    return {
        "time_s": raw if source == "et_scene" else relative,
        "uncertainty_s": uncertainty,
        "confidence": _text(annotation["confidence"]),
    }


def _set_mapping_coefficients(
    row: dict[str, object], intercept: float, scale: float
) -> None:
    row["driver_to_et_raw_intercept_s"] = intercept
    row["driver_to_et_raw_scale"] = scale
    row["driver_to_et_raw_drift_ppm"] = (scale - 1.0) * 1_000_000.0


def build_driver_et_alignments(
    annotations: pd.DataFrame,
    *,
    video_pairs: pd.DataFrame | None = None,
    drift_warning_ppm: float = 2_000.0,
    drift_reject_ppm: float = 10_000.0,
    max_pair_uncertainty_s: float = 0.5,
    offset_only_drift_bound_ppm: float = 5_000.0,
) -> pd.DataFrame:
    for name, value in {
        "drift_warning_ppm": drift_warning_ppm,
        "drift_reject_ppm": drift_reject_ppm,
        "max_pair_uncertainty_s": max_pair_uncertainty_s,
        "offset_only_drift_bound_ppm": offset_only_drift_bound_ppm,
    }.items():
        value = _finite_float(value, name)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if drift_reject_ppm <= drift_warning_ppm:
        raise ValueError("drift_reject_ppm must exceed drift_warning_ppm")
    if offset_only_drift_bound_ppm >= 1_000_000.0:
        raise ValueError(
            "offset_only_drift_bound_ppm must be below 1,000,000 "
            "so its scale bound stays positive"
        )

    required = set(panel.ANNOTATION_COLUMNS)
    missing_columns = sorted(required.difference(annotations.columns))
    if missing_columns:
        raise ValueError(
            f"annotation table is missing columns: {missing_columns}"
        )

    accepted = annotations.loc[annotations["status"].astype(str).eq("accepted")].copy()

    keys: set[tuple[str, str, str, str]] = set()
    if video_pairs is not None and not video_pairs.empty:
        missing_job_columns = sorted(
            {"project", "pid", "visit"}.difference(video_pairs.columns)
        )
        if missing_job_columns:
            raise ValueError(
                f"video_pairs is missing columns: {missing_job_columns}"
            )
        for _, job in video_pairs.iterrows():
            job_key = _group_key(
                [job["project"], job["pid"], job["visit"], "main"]
            )
            keys.add(job_key)
    for values in annotations[GROUP_COLUMNS].itertuples(
        index=False, name=None
    ):
        keys.add(_group_key(values))
    if not keys:
        return pd.DataFrame(columns=ALIGNMENT_COLUMNS)

    rows: list[dict[str, object]] = []
    for key in sorted(keys):
        row = _base_row(key)
        invalid_group = False
        warning_messages: list[str] = []
        group_mask = pd.Series(True, index=accepted.index)
        for column, value in zip(GROUP_COLUMNS, key):
            group_mask &= accepted[column].astype(str).eq(value)
        group = accepted.loc[group_mask]

        source_rows: dict[tuple[str, str], pd.Series] = {}
        for _, annotation in group.iterrows():
            source_rows[
                (_text(annotation["source"]), _text(annotation["event"]))
            ] = annotation

        anchors: dict[str, dict[str, dict[str, object]]] = {}
        for event in EVENTS:
            anchors[event] = {}
            for source in ("driver", "et_scene"):
                annotation = source_rows.get((source, event))
                if annotation is None:
                    continue
                try:
                    anchor = _anchor_from_row(annotation, source=source)
                    anchors[event][source] = anchor
                except (KeyError, TypeError, ValueError):
                    invalid_group = True

        paired = [
            event
            for event in EVENTS
            if {"driver", "et_scene"}.issubset(anchors[event])
        ]
        row["anchor_count"] = len(paired)

        if len(paired) == 1:
            event = paired[0]
            driver_anchor = anchors[event]["driver"]
            et_anchor = anchors[event]["et_scene"]
            intercept = float(et_anchor["time_s"]) - float(
                driver_anchor["time_s"]
            )
            _set_mapping_coefficients(row, intercept, 1.0)
            row["driver_to_et_raw_drift_ppm"] = np.nan
            drift_fraction = offset_only_drift_bound_ppm / 1_000_000.0
            pair_uncertainty = float(et_anchor["uncertainty_s"]) + (
                1.0 + drift_fraction
            ) * float(driver_anchor["uncertainty_s"])
            row["mapping_status"] = "offset_only"
            row["usable"] = True
            row["quality_grade"] = "C"
            row["review_required"] = True
            row["mapping_reason"] = (
                "one paired event; usable with scale fixed to 1, "
                "no measured drift, "
                "and a growing drift-bound uncertainty"
            )
            row["driver_interpolation_start_s"] = driver_anchor["time_s"]
            row["driver_interpolation_end_s"] = driver_anchor["time_s"]
            warning = (
                "single anchor cannot estimate clock drift; "
                "scale is fixed to 1 and "
                "uncertainty grows using the configured drift bound"
            )
            warning_messages.append(warning)
            if pair_uncertainty > max_pair_uncertainty_s:
                message = (
                    f"anchor-pair uncertainty {pair_uncertainty:.6g} s "
                    "exceeds "
                    f"{max_pair_uncertainty_s:.6g} s"
                )
                warning_messages.append(message)
            if "low" in {
                str(driver_anchor["confidence"]),
                str(et_anchor["confidence"]),
            }:
                message = "at least one accepted anchor has low confidence"
                warning_messages.append(message)
        elif len(paired) == 2:
            off_driver = anchors["panel_off"]["driver"]
            on_driver = anchors["panel_on"]["driver"]
            off_et = anchors["panel_off"]["et_scene"]
            on_et = anchors["panel_on"]["et_scene"]
            driver_off = float(off_driver["time_s"])
            driver_on = float(on_driver["time_s"])
            et_off = float(off_et["time_s"])
            et_on = float(on_et["time_s"])
            driver_span = driver_on - driver_off
            et_span = et_on - et_off
            row["driver_interpolation_start_s"] = driver_off
            row["driver_interpolation_end_s"] = driver_on
            if driver_span <= 0 or et_span <= 0:
                invalid_group = True
            else:
                scale = et_span / driver_span
                intercept = et_off - scale * driver_off
                _set_mapping_coefficients(row, intercept, scale)
                pair_uncertainties: list[float] = []
                confidences: list[str] = []
                for driver_anchor, et_anchor in ((off_driver, off_et), (on_driver, on_et)):
                    pair_uncertainty = float(et_anchor["uncertainty_s"]) + abs(
                        scale
                    ) * float(driver_anchor["uncertainty_s"])
                    pair_uncertainties.append(pair_uncertainty)
                    confidences.extend(
                        [
                            str(driver_anchor["confidence"]),
                            str(et_anchor["confidence"]),
                        ]
                    )
                if driver_off + float(
                    off_driver["uncertainty_s"]
                ) >= driver_on - float(
                    on_driver["uncertainty_s"]
                ) or et_off + float(
                    off_et["uncertainty_s"]
                ) >= et_on - float(
                    on_et["uncertainty_s"]
                ):
                    invalid_group = True
                maximum_uncertainty = max(pair_uncertainties)
                drift = abs(float(row["driver_to_et_raw_drift_ppm"]))
                review_reasons: list[str] = []
                if drift > drift_reject_ppm:
                    invalid_group = True
                elif drift > drift_warning_ppm:
                    message = (
                        f"absolute clock drift {drift:.1f} ppm "
                        "exceeds review threshold "
                        f"{drift_warning_ppm:.1f} ppm"
                    )
                    review_reasons.append(message)
                if maximum_uncertainty > max_pair_uncertainty_s:
                    message = (
                        f"anchor-pair uncertainty {maximum_uncertainty:.6g} s "
                        "exceeds "
                        f"{max_pair_uncertainty_s:.6g} s"
                    )
                    review_reasons.append(message)
                if "low" in confidences:
                    message = "at least one accepted anchor has low confidence"
                    review_reasons.append(message)

                if invalid_group:
                    row["mapping_status"] = "invalid_affine"
                    row["quality_grade"] = "D"
                    row["review_required"] = True
                    row["mapping_reason"] = (
                        "affine coefficients are diagnostic only; "
                        "hard QC failed"
                    )
                elif review_reasons:
                    row["mapping_status"] = "review_affine"
                    row["usable"] = True
                    row["quality_grade"] = "C"
                    row["review_required"] = True
                    row["mapping_reason"] = "; ".join(review_reasons)
                    warning_messages.extend(review_reasons)
                else:
                    row["mapping_status"] = "affine"
                    row["usable"] = True
                    row["quality_grade"] = (
                        "A"
                        if all(
                            confidence == "high" for confidence in confidences
                        )
                        and drift <= 500.0
                        and maximum_uncertainty
                        <= min(0.25, max_pair_uncertainty_s)
                        else "B"
                    )
                    row["mapping_reason"] = (
                        "two accepted event pairs passed mapping QC"
                    )

        if invalid_group:
            row["usable"] = False
            if row["mapping_status"] == "offset_only":
                row["mapping_status"] = "invalid_offset_only"
                row["quality_grade"] = "D"
                row["review_required"] = True
                row["mapping_reason"] = (
                    "offset-only coefficients are diagnostic only; "
                    "hard QC failed"
                )
            elif row["mapping_status"] == "unavailable":
                row["mapping_status"] = "invalid"
                row["review_required"] = True
                row["mapping_reason"] = (
                    "annotation or ET-recording validation failed"
                )
        row["warnings"] = " | ".join(dict.fromkeys(warning_messages))
        rows.append(row)

    return pd.DataFrame(
        rows, columns=ALIGNMENT_COLUMNS, dtype=object
    ).sort_values(GROUP_COLUMNS, ignore_index=True)


def save_driver_et_alignment_results(
    alignments: pd.DataFrame, *, alignment_csv: str | Path
) -> Path:
    if alignments.empty:
        raise ValueError(
            "alignment table is empty; there is no discovered visit to save"
        )
    path = Path(alignment_csv).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    alignments.loc[:, ALIGNMENT_COLUMNS].to_csv(path, index=False)
    return path


__all__ = [
    "ALIGNMENT_COLUMNS",
    "build_driver_et_alignments",
    "save_driver_et_alignment_results",
]
