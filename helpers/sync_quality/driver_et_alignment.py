r"""Build post-hoc Driver-video to Tobii ET-raw clock mappings.

The manual annotations retain only the exact PTS bracket and its time base.
Driver anchors are relative to the first displayed frame; Tobii G3 scene-video
absolute PTS values are already on the ET-raw recording clock.

Two matched physical transitions determine an affine mapping.  One transition
can only determine a constant offset, so that mapping fixes scale to one and
reports an uncertainty envelope that grows away from the anchor under an
explicit clock-drift bound.  Soft QC warnings remain visible without discarding
an otherwise valid mapping.  Two points always have zero fitted residual, so
residual is intentionally not reported as alignment-quality evidence.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import panel_annotation as panel


ALIGNMENT_ALGORITHM = "driver_et_panel_affine_v2"
GROUP_COLUMNS = ["project", "pid", "visit", "segment_id"]
EVENTS = ("panel_off", "panel_on")

ISSUE_COLUMNS = [
    *GROUP_COLUMNS,
    "severity",
    "code",
    "source",
    "event",
    "message",
]


ALIGNMENT_COLUMNS = [
    "algorithm",
    *GROUP_COLUMNS,
    "driver_time_reference",
    "et_raw_time_reference",
    "mapping_equation",
    "et_scene_to_et_raw_rule",
    "video_discovery_status",
    "driver_candidate_count",
    "et_scene_candidate_count",
    "mapping_status",
    "fit_available",
    "usable",
    "quality_grade",
    "review_required",
    "mapping_reason",
    "anchor_count",
    "anchor_events",
    "driver_to_et_raw_intercept_s",
    "driver_to_et_raw_scale",
    "driver_to_et_raw_drift_ppm",
    "et_raw_to_driver_intercept_s",
    "et_raw_to_driver_scale",
    "driver_anchor_span_s",
    "et_raw_anchor_span_s",
    "anchor_span_difference_s",
    "panel_off_offset_s",
    "panel_on_offset_s",
    "offset_change_s",
    "driver_interpolation_start_s",
    "driver_interpolation_end_s",
    "et_raw_interpolation_start_s",
    "et_raw_interpolation_end_s",
    "max_anchor_pair_uncertainty_s",
    "uncertainty_model",
    "offset_only_drift_bound_ppm",
    "drift_warning_ppm",
    "drift_reject_ppm",
    "max_pair_uncertainty_threshold_s",
    "driver_video_path",
    "et_scene_video_path",
    "et_scene_pts_origin_s",
    "warnings",
]

for _event in EVENTS:
    ALIGNMENT_COLUMNS.extend(
        [
            f"driver_{_event}_s",
            f"driver_{_event}_uncertainty_s",
            f"driver_{_event}_confidence",
            f"et_raw_{_event}_s",
            f"et_scene_{_event}_relative_s",
            f"et_scene_{_event}_pts_origin_s",
            f"et_raw_{_event}_uncertainty_s",
            f"et_scene_{_event}_confidence",
            f"{_event}_pair_uncertainty_s",
        ]
    )


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


def _issue(
    issues: list[dict[str, object]],
    key: Sequence[str],
    severity: str,
    code: str,
    message: str,
    *,
    source: str = "",
    event: str = "",
) -> None:
    issues.append(
        {
            **dict(zip(GROUP_COLUMNS, key)),
            "severity": severity,
            "code": code,
            "source": source,
            "event": event,
            "message": message,
        }
    )


def _base_row(
    key: Sequence[str],
    *,
    drift_warning_ppm: float,
    drift_reject_ppm: float,
    max_pair_uncertainty_s: float,
    offset_only_drift_bound_ppm: float,
) -> dict[str, object]:
    row: dict[str, object] = {column: np.nan for column in ALIGNMENT_COLUMNS}
    row.update(
        {
            "algorithm": ALIGNMENT_ALGORITHM,
            **dict(zip(GROUP_COLUMNS, key)),
            "driver_time_reference": "seconds_from_first_displayed_driver_frame",
            "et_raw_time_reference": "seconds_from_tobii_recording_start",
            "mapping_equation": "et_raw_s = intercept_s + scale * driver_relative_s",
            "et_scene_to_et_raw_rule": "scene_relative_midpoint_s + scene_pts_origin_s",
            "video_discovery_status": "",
            "mapping_status": "unavailable",
            "fit_available": False,
            "usable": False,
            "quality_grade": "D",
            "review_required": False,
            "mapping_reason": "no paired accepted panel anchors",
            "anchor_count": 0,
            "anchor_events": "",
            "uncertainty_model": "unavailable",
            "offset_only_drift_bound_ppm": float(offset_only_drift_bound_ppm),
            "drift_warning_ppm": float(drift_warning_ppm),
            "drift_reject_ppm": float(drift_reject_ppm),
            "max_pair_uncertainty_threshold_s": float(max_pair_uncertainty_s),
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


def _accepted_rows(annotations: pd.DataFrame) -> pd.DataFrame:
    accepted = annotations.copy()
    if accepted.empty:
        return accepted
    mask = accepted["status"].astype(str).eq("accepted")
    return accepted.loc[mask].copy()


def _anchor_from_row(
    annotation: pd.Series,
    *,
    source: str,
) -> dict[str, object]:
    relative, raw, uncertainty, origin = panel.annotation_bracket_values(annotation)
    confidence = _text(annotation["confidence"])
    result: dict[str, object] = {
        "time_s": relative,
        "uncertainty_s": uncertainty,
        "confidence": confidence,
    }
    if source == "et_scene":
        result.update(
            {
                "relative_s": relative,
                "pts_origin_s": origin,
                "time_s": raw,
            }
        )
    return result


def _set_mapping_coefficients(
    row: dict[str, object], intercept: float, scale: float
) -> None:
    row["driver_to_et_raw_intercept_s"] = intercept
    row["driver_to_et_raw_scale"] = scale
    row["driver_to_et_raw_drift_ppm"] = (scale - 1.0) * 1_000_000.0
    row["et_raw_to_driver_scale"] = 1.0 / scale
    row["et_raw_to_driver_intercept_s"] = -intercept / scale


def build_driver_et_alignments(
    annotations: pd.DataFrame,
    *,
    video_pairs: pd.DataFrame | None = None,
    drift_warning_ppm: float = 2_000.0,
    drift_reject_ppm: float = 10_000.0,
    max_pair_uncertainty_s: float = 0.5,
    offset_only_drift_bound_ppm: float = 5_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one auditable Driver-relative -> ET-raw mapping per visit.

    ``usable`` means that coefficients are mathematically available and no hard
    validation rule failed.  Soft warnings are carried separately in
    ``review_required``, ``quality_grade``, ``mapping_reason`` and the issue
    table; they do not discard an otherwise valid mapping.
    """

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
            "offset_only_drift_bound_ppm must be below 1,000,000 so its scale bound stays positive"
        )

    required = set(panel.ANNOTATION_COLUMNS)
    missing_columns = sorted(required.difference(annotations.columns))
    if missing_columns:
        raise ValueError(f"annotation table is missing columns: {missing_columns}")

    issues: list[dict[str, object]] = []
    invalid_groups: set[tuple[str, str, str, str]] = set()
    accepted = _accepted_rows(annotations)

    keys: set[tuple[str, str, str, str]] = set()
    jobs_by_key: dict[tuple[str, str, str, str], pd.Series] = {}
    if video_pairs is not None and not video_pairs.empty:
        missing_job_columns = sorted(
            {"project", "pid", "visit"}.difference(video_pairs.columns)
        )
        if missing_job_columns:
            raise ValueError(f"video_pairs is missing columns: {missing_job_columns}")
        for _, job in video_pairs.iterrows():
            job_key = _group_key([job["project"], job["pid"], job["visit"], "main"])
            keys.add(job_key)
            jobs_by_key[job_key] = job
    for values in annotations[GROUP_COLUMNS].itertuples(index=False, name=None):
        keys.add(_group_key(values))
    if not keys:
        return (
            pd.DataFrame(columns=ALIGNMENT_COLUMNS),
            pd.DataFrame(columns=ISSUE_COLUMNS),
        )

    rows: list[dict[str, object]] = []
    for key in sorted(keys):
        row = _base_row(
            key,
            drift_warning_ppm=drift_warning_ppm,
            drift_reject_ppm=drift_reject_ppm,
            max_pair_uncertainty_s=max_pair_uncertainty_s,
            offset_only_drift_bound_ppm=offset_only_drift_bound_ppm,
        )
        job = jobs_by_key.get(key)
        if job is not None:
            row["video_discovery_status"] = _text(job.get("discovery_status"))
            row["driver_candidate_count"] = job.get("driver_candidate_count", np.nan)
            row["et_scene_candidate_count"] = job.get("et_scene_candidate_count", np.nan)
            if not _missing(job.get("driver_path")):
                row["driver_video_path"] = _text(job.get("driver_path"))
            if not _missing(job.get("et_scene_path")):
                row["et_scene_video_path"] = _text(job.get("et_scene_path"))
        warning_messages: list[str] = []
        group_mask = pd.Series(True, index=accepted.index)
        for column, value in zip(GROUP_COLUMNS, key):
            group_mask &= accepted[column].astype(str).eq(value)
        group = accepted.loc[group_mask]

        source_rows: dict[tuple[str, str], pd.Series] = {}
        for _, annotation in group.iterrows():
            source_rows[(_text(annotation["source"]), _text(annotation["event"]))] = annotation

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
                    if source == "driver":
                        row[f"driver_{event}_s"] = anchor["time_s"]
                        row[f"driver_{event}_uncertainty_s"] = anchor["uncertainty_s"]
                        row[f"driver_{event}_confidence"] = anchor["confidence"]
                    else:
                        row["et_scene_pts_origin_s"] = anchor["pts_origin_s"]
                        row[f"et_raw_{event}_s"] = anchor["time_s"]
                        row[f"et_scene_{event}_relative_s"] = anchor["relative_s"]
                        row[f"et_scene_{event}_pts_origin_s"] = anchor["pts_origin_s"]
                        row[f"et_raw_{event}_uncertainty_s"] = anchor["uncertainty_s"]
                        row[f"et_scene_{event}_confidence"] = anchor["confidence"]
                except (KeyError, TypeError, ValueError) as error:
                    invalid_groups.add(key)
                    _issue(
                        issues,
                        key,
                        "error",
                        "invalid_anchor",
                        str(error),
                        source=source,
                        event=event,
                    )

        paired = [
            event
            for event in EVENTS
            if {"driver", "et_scene"}.issubset(anchors[event])
        ]
        row["anchor_count"] = len(paired)
        row["anchor_events"] = "|".join(paired)

        if not paired:
            _issue(
                issues,
                key,
                "warning",
                "no_paired_anchor",
                "no panel event has accepted annotations in both Driver and ET scene",
            )
        elif len(paired) == 1:
            event = paired[0]
            driver_anchor = anchors[event]["driver"]
            et_anchor = anchors[event]["et_scene"]
            intercept = float(et_anchor["time_s"]) - float(driver_anchor["time_s"])
            _set_mapping_coefficients(row, intercept, 1.0)
            # Scale=1 is an explicit working assumption, not an observed zero
            # drift.  Keep the coefficient usable while leaving measured drift
            # undefined so downstream summaries cannot mistake it for evidence.
            row["driver_to_et_raw_drift_ppm"] = np.nan
            drift_fraction = offset_only_drift_bound_ppm / 1_000_000.0
            pair_uncertainty = float(et_anchor["uncertainty_s"]) + (
                1.0 + drift_fraction
            ) * float(driver_anchor["uncertainty_s"])
            row[f"{event}_pair_uncertainty_s"] = pair_uncertainty
            row["max_anchor_pair_uncertainty_s"] = pair_uncertainty
            row["fit_available"] = True
            row["mapping_status"] = "offset_only"
            row["usable"] = True
            row["quality_grade"] = "C"
            row["review_required"] = True
            row["mapping_reason"] = (
                "one paired event; usable with scale fixed to 1, no measured drift, "
                "and a growing drift-bound uncertainty"
            )
            row["uncertainty_model"] = "bounded_single_anchor_plus_drift_assumption"
            row["panel_off_offset_s" if event == "panel_off" else "panel_on_offset_s"] = intercept
            row["driver_interpolation_start_s"] = driver_anchor["time_s"]
            row["driver_interpolation_end_s"] = driver_anchor["time_s"]
            row["et_raw_interpolation_start_s"] = et_anchor["time_s"]
            row["et_raw_interpolation_end_s"] = et_anchor["time_s"]
            warning = (
                "single anchor cannot estimate clock drift; scale is fixed to 1 and "
                "uncertainty grows using the configured drift bound"
            )
            warning_messages.append(warning)
            _issue(issues, key, "warning", "single_anchor_offset_only", warning, event=event)
            if pair_uncertainty > max_pair_uncertainty_s:
                message = (
                    f"anchor-pair uncertainty {pair_uncertainty:.6g} s exceeds "
                    f"{max_pair_uncertainty_s:.6g} s"
                )
                warning_messages.append(message)
                _issue(
                    issues,
                    key,
                    "warning",
                    "anchor_uncertainty_requires_review",
                    message,
                    event=event,
                )
            if "low" in {
                str(driver_anchor["confidence"]),
                str(et_anchor["confidence"]),
            }:
                message = "at least one accepted anchor has low confidence"
                warning_messages.append(message)
                _issue(
                    issues,
                    key,
                    "warning",
                    "low_confidence_anchor",
                    message,
                    event=event,
                )
        else:
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
            row["driver_anchor_span_s"] = driver_span
            row["et_raw_anchor_span_s"] = et_span
            row["anchor_span_difference_s"] = et_span - driver_span
            row["panel_off_offset_s"] = et_off - driver_off
            row["panel_on_offset_s"] = et_on - driver_on
            row["offset_change_s"] = (et_on - driver_on) - (et_off - driver_off)
            row["driver_interpolation_start_s"] = driver_off
            row["driver_interpolation_end_s"] = driver_on
            row["et_raw_interpolation_start_s"] = et_off
            row["et_raw_interpolation_end_s"] = et_on
            if driver_span <= 0 or et_span <= 0:
                invalid_groups.add(key)
                _issue(
                    issues,
                    key,
                    "error",
                    "invalid_anchor_order",
                    "panel_off must precede panel_on in both time domains",
                )
            else:
                scale = et_span / driver_span
                intercept = et_off - scale * driver_off
                _set_mapping_coefficients(row, intercept, scale)
                row["fit_available"] = True
                row["uncertainty_model"] = "bounded_linear_two_anchor_interpolation"
                pair_uncertainties: list[float] = []
                confidences: list[str] = []
                for event, driver_anchor, et_anchor in (
                    ("panel_off", off_driver, off_et),
                    ("panel_on", on_driver, on_et),
                ):
                    pair_uncertainty = float(et_anchor["uncertainty_s"]) + abs(scale) * float(
                        driver_anchor["uncertainty_s"]
                    )
                    row[f"{event}_pair_uncertainty_s"] = pair_uncertainty
                    pair_uncertainties.append(pair_uncertainty)
                    confidences.extend(
                        [str(driver_anchor["confidence"]), str(et_anchor["confidence"])]
                    )
                if (
                    driver_off + float(off_driver["uncertainty_s"])
                    >= driver_on - float(on_driver["uncertainty_s"])
                    or et_off + float(off_et["uncertainty_s"])
                    >= et_on - float(on_et["uncertainty_s"])
                ):
                    invalid_groups.add(key)
                    _issue(
                        issues,
                        key,
                        "error",
                        "overlapping_anchor_brackets",
                        "panel-off and panel-on uncertainty intervals overlap, so affine uncertainty is unbounded",
                    )
                maximum_uncertainty = max(pair_uncertainties)
                row["max_anchor_pair_uncertainty_s"] = maximum_uncertainty
                drift = abs(float(row["driver_to_et_raw_drift_ppm"]))
                review_reasons: list[str] = []
                hard_drift = drift > drift_reject_ppm
                if hard_drift:
                    message = (
                        f"absolute clock drift {drift:.1f} ppm exceeds reject threshold "
                        f"{drift_reject_ppm:.1f} ppm"
                    )
                    invalid_groups.add(key)
                    _issue(issues, key, "error", "drift_exceeds_reject_threshold", message)
                elif drift > drift_warning_ppm:
                    message = (
                        f"absolute clock drift {drift:.1f} ppm exceeds review threshold "
                        f"{drift_warning_ppm:.1f} ppm"
                    )
                    review_reasons.append(message)
                    _issue(issues, key, "warning", "drift_requires_review", message)
                if maximum_uncertainty > max_pair_uncertainty_s:
                    message = (
                        f"anchor-pair uncertainty {maximum_uncertainty:.6g} s exceeds "
                        f"{max_pair_uncertainty_s:.6g} s"
                    )
                    review_reasons.append(message)
                    _issue(issues, key, "warning", "anchor_uncertainty_requires_review", message)
                if "low" in confidences:
                    message = "at least one accepted anchor has low confidence"
                    review_reasons.append(message)
                    _issue(issues, key, "warning", "low_confidence_anchor", message)

                if key in invalid_groups:
                    row["mapping_status"] = "invalid_affine"
                    row["quality_grade"] = "D"
                    row["review_required"] = True
                    row["mapping_reason"] = "affine coefficients are diagnostic only; hard QC failed"
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
                        if all(confidence == "high" for confidence in confidences)
                        and drift <= 500.0
                        and maximum_uncertainty <= min(0.25, max_pair_uncertainty_s)
                        else "B"
                    )
                    row["mapping_reason"] = "two accepted event pairs passed mapping QC"

        if key in invalid_groups:
            row["usable"] = False
            if row["mapping_status"] == "offset_only":
                row["mapping_status"] = "invalid_offset_only"
                row["quality_grade"] = "D"
                row["review_required"] = True
                row["mapping_reason"] = (
                    "offset-only coefficients are diagnostic only; hard QC failed"
                )
            elif row["mapping_status"] == "unavailable":
                row["mapping_status"] = "invalid"
                row["review_required"] = True
                row["mapping_reason"] = "annotation or ET-recording validation failed"
        row["warnings"] = " | ".join(dict.fromkeys(warning_messages))
        rows.append(row)

    mappings = pd.DataFrame(
        rows, columns=ALIGNMENT_COLUMNS, dtype=object
    ).sort_values(
        GROUP_COLUMNS, ignore_index=True
    )
    issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    if not issue_frame.empty:
        issue_frame = issue_frame.sort_values(
            [*GROUP_COLUMNS, "severity", "code"], ignore_index=True
        )
    return mappings, issue_frame


def save_driver_et_alignment_results(
    alignments: pd.DataFrame,
    issues: pd.DataFrame,
    *,
    alignment_csv: str | Path,
    issues_csv: str | Path,
) -> tuple[Path, Path]:
    """Write the current alignment and issue tables directly to CSV."""
    alignments = alignments.loc[:, ALIGNMENT_COLUMNS]
    issues = issues.loc[:, ISSUE_COLUMNS]
    if alignments.empty:
        raise ValueError("alignment table is empty; there is no discovered visit to save")
    alignment_path = Path(alignment_csv).expanduser().resolve()
    issue_path = Path(issues_csv).expanduser().resolve()
    if alignment_path == issue_path:
        raise ValueError("alignment_csv and issues_csv must be different files")
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    issue_path.parent.mkdir(parents=True, exist_ok=True)
    alignments.to_csv(alignment_path, index=False)
    issues.to_csv(issue_path, index=False)
    return alignment_path, issue_path


__all__ = [
    "ALIGNMENT_ALGORITHM",
    "ALIGNMENT_COLUMNS",
    "ISSUE_COLUMNS",
    "build_driver_et_alignments",
    "save_driver_et_alignment_results",
]
