r"""
Manual speedometer-panel transition annotations for ET scene and Driver video.

This module deliberately does not infer a timestamp from FPS.  It asks
``ffprobe`` for every displayed frame's integer presentation timestamp (PTS),
normalises those timestamps to the first displayed frame, and lets an operator
bracket each physical panel transition with:

* the last frame that is clearly in the old state; and
* the first frame that is clearly in the new state.

The midpoint is the manual anchor and half of the bracket width is its stated
uncertainty.  ``panel_off`` means on -> off and ``panel_on`` means off -> on.
No clock mapping or synchronization-quality grade is calculated here.

A notebook
widget is provided as a convenience, but ipywidgets is imported only when that
widget is launched and is therefore not a hard module dependency.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd


SOURCES = ("driver", "et_scene")
EVENTS = ("panel_off", "panel_on")
ANNOTATION_STATUSES = ("draft", "accepted", "ambiguous", "not_visible")
CONFIDENCE_LEVELS = ("high", "medium", "low", "not_rated")
VIDEO_EXTENSIONS = frozenset(
    {".mkv", ".mp4", ".m4v", ".mov", ".avi", ".webm"}
)

PAIR_COLUMNS = [
    "project",
    "pid",
    "visit",
    "driver_path",
    "et_scene_path",
    "driver_candidate_count",
    "et_scene_candidate_count",
    "driver_candidates",
    "et_scene_candidates",
    "discovery_status",
]

ANNOTATION_KEY_COLUMNS = [
    "project",
    "pid",
    "visit",
    "segment_id",
    "source",
    "event",
]

ANNOTATION_COLUMNS = [
    *ANNOTATION_KEY_COLUMNS,
    "status",
    "confidence",
    "notes",
    "pts_time_base_num",
    "pts_time_base_den",
    "pts_origin_tick",
    "last_old_pts_tick",
    "first_new_pts_tick",
]


@dataclass(frozen=True)
class VideoFrameIndex:
    """Presentation-order frame index made from a video's real PTS values."""

    video_path: Path
    pts_ticks: np.ndarray
    relative_times_s: np.ndarray
    key_frames: np.ndarray
    time_base_num: int
    time_base_den: int
    pts_origin_tick: int
    width: int
    height: int

    @property
    def frame_count(self) -> int:
        return int(self.pts_ticks.size)

    @property
    def duration_s(self) -> float:
        return float(self.relative_times_s[-1])

    @property
    def pts_origin_s(self) -> float:
        return self.pts_origin_tick * self.time_base_num / self.time_base_den

def _parse_pid_visit(value: str) -> tuple[str, str] | None:
    """Read pid/visit from the first two underscore-delimited name tokens."""
    name = Path(value).name
    suffix = Path(name).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        name = name[: -len(suffix)]
    parts = name.split("_")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _normalise_filters(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}


def discover_panel_video_pairs(
    data_root: str | Path,
    *,
    project: str,
    participants: Iterable[str] | None = None,
    visits: Iterable[str] | None = None,
    driver_subfolder: str = "Driver Video",
    et_subfolder: str = "ET",
    scene_video_name: str = "scenevideo.mp4",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Discover Driver and ET scene videos and outer-join them on ``(pid, visit)``.

    A side is usable only when exactly one candidate was found.  Multiple
    candidates remain visible in the result and are never resolved by taking
    the first directory entry.  ET scene videos must be directly inside an ET
    recording directory; loose files and ``*_aborted`` directories without a
    scene video therefore cannot be selected accidentally.
    """
    root = Path(data_root).expanduser()
    participant_filter = _normalise_filters(participants)
    visit_filter = _normalise_filters(visits)

    if not root.is_dir():
        if verbose:
            print(f"Data root is unavailable: {root}")
        result = pd.DataFrame(columns=PAIR_COLUMNS)
        result.attrs["data_root_error"] = f"Data root is unavailable: {root}"
        return result

    driver_by_key: dict[tuple[str, str], list[Path]] = {}
    scene_by_key: dict[tuple[str, str], list[Path]] = {}

    for participant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        pid_from_folder = participant_dir.name
        if participant_filter is not None and pid_from_folder not in participant_filter:
            continue

        driver_dir = participant_dir / driver_subfolder
        if driver_dir.is_dir():
            for video_path in sorted(path for path in driver_dir.rglob("*") if path.is_file()):
                if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                key = _parse_pid_visit(video_path.name)
                if key is None or key[0] != pid_from_folder:
                    continue
                if visit_filter is not None and key[1] not in visit_filter:
                    continue
                driver_by_key.setdefault(key, []).append(video_path.resolve())

        et_dir = participant_dir / et_subfolder
        if et_dir.is_dir():
            for recording_dir in sorted(path for path in et_dir.iterdir() if path.is_dir()):
                scene_path = recording_dir / scene_video_name
                if not scene_path.is_file():
                    continue
                key = _parse_pid_visit(recording_dir.name)
                if key is None or key[0] != pid_from_folder:
                    continue
                if visit_filter is not None and key[1] not in visit_filter:
                    continue
                scene_by_key.setdefault(key, []).append(scene_path.resolve())

    rows: list[dict[str, object]] = []
    for pid, visit in sorted(set(driver_by_key) | set(scene_by_key)):
        driver_candidates = sorted(set(driver_by_key.get((pid, visit), [])))
        scene_candidates = sorted(set(scene_by_key.get((pid, visit), [])))
        driver_count = len(driver_candidates)
        scene_count = len(scene_candidates)

        problems: list[str] = []
        if driver_count == 0:
            problems.append("missing_driver")
        elif driver_count > 1:
            problems.append("ambiguous_driver")
        if scene_count == 0:
            problems.append("missing_et_scene")
        elif scene_count > 1:
            problems.append("ambiguous_et_scene")

        rows.append(
            {
                "project": str(project),
                "pid": pid,
                "visit": visit,
                "driver_path": str(driver_candidates[0]) if driver_count == 1 else pd.NA,
                "et_scene_path": str(scene_candidates[0]) if scene_count == 1 else pd.NA,
                "driver_candidate_count": driver_count,
                "et_scene_candidate_count": scene_count,
                "driver_candidates": " | ".join(str(path) for path in driver_candidates),
                "et_scene_candidates": " | ".join(str(path) for path in scene_candidates),
                "discovery_status": "ready" if not problems else ";".join(problems),
            }
        )

    result = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    if verbose:
        ready = int((result["discovery_status"] == "ready").sum()) if not result.empty else 0
        print(f"Discovered {len(result)} visit(s): {ready} ready for paired annotation.")
        if len(result) != ready:
            print("Missing or ambiguous videos remain in the discovery table for review.")
    return result


def _resolve_ffprobe(ffprobe_path: str | Path | None) -> str:
    if ffprobe_path is not None:
        candidate = Path(ffprobe_path).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        resolved = shutil.which(str(ffprobe_path))
        if resolved:
            return resolved
        raise FileNotFoundError(f"ffprobe was not found at: {ffprobe_path}")

    resolved = shutil.which("ffprobe")
    if resolved:
        return resolved

    raise FileNotFoundError(
        "ffprobe is required to read real video PTS values. Add it to PATH."
    )


def _resolve_ffmpeg(ffmpeg_path: str | Path | None) -> str:
    """Resolve the companion decoder used for exact random PTS seeks."""
    if ffmpeg_path is not None:
        candidate = Path(ffmpeg_path).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        resolved = shutil.which(str(ffmpeg_path))
        if resolved:
            return resolved
        raise FileNotFoundError(f"ffmpeg was not found at: {ffmpeg_path}")

    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved

    raise FileNotFoundError(
        "ffmpeg is required when a video container cannot seek by frame number. "
        "Add it to PATH."
    )


def _fraction_from_time_base(value: str) -> tuple[int, int]:
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"Invalid video time_base reported by ffprobe: {value!r}") from error
    if fraction <= 0:
        raise ValueError(f"Video time_base must be positive, got {value!r}")
    return fraction.numerator, fraction.denominator


def _run_ffprobe_json(
    command: Sequence[str],
    *,
    video_path: Path,
    timeout_s: float,
) -> dict[str, object]:
    """Run one ffprobe JSON query with consistent timeout/error handling."""
    run_kwargs: dict[str, object] = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "check": False,
        "timeout": timeout_s,
    }
    if os.name == "nt":
        run_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        completed = subprocess.run(list(command), **run_kwargs)
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"ffprobe did not finish indexing {video_path} within {timeout_s:g} s"
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown ffprobe error").strip()
        raise RuntimeError(f"ffprobe could not index {video_path}: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"ffprobe returned invalid JSON for {video_path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"ffprobe returned invalid JSON for {video_path}")
    return payload


def _make_video_frame_index(
    *,
    video_path: Path,
    stream: Mapping[str, object],
    pts_values: Sequence[int],
    key_frame_values: Sequence[bool],
) -> VideoFrameIndex:
    """Validate common metadata/arrays and construct an immutable PTS index."""
    time_base_num, time_base_den = _fraction_from_time_base(
        str(stream.get("time_base", ""))
    )
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"ffprobe returned invalid dimensions for {video_path}")
    if not pts_values:
        raise ValueError(f"No displayed video frames were found in {video_path}")
    if len(pts_values) != len(key_frame_values):
        raise ValueError(f"PTS/key-frame array length mismatch in {video_path}")

    pts_ticks = np.asarray(pts_values, dtype=np.int64)
    differences = np.diff(pts_ticks)
    if np.any(differences <= 0):
        bad_index = int(np.flatnonzero(differences <= 0)[0])
        relationship = "duplicate" if differences[bad_index] == 0 else "backward"
        raise ValueError(
            f"Video PTS values are not strictly increasing ({relationship} at frames "
            f"{bad_index}/{bad_index + 1}) in {video_path}. The video cannot be used "
            "for an accepted manual anchor without repairing its timestamps."
        )

    origin = int(pts_ticks[0])
    relative_times_s = (
        (pts_ticks - origin).astype(np.float64) * time_base_num / time_base_den
    )
    key_frames = np.asarray(key_frame_values, dtype=np.bool_)
    if not key_frames.any():
        key_frames[0] = True
    pts_ticks.setflags(write=False)
    relative_times_s.setflags(write=False)
    key_frames.setflags(write=False)
    return VideoFrameIndex(
        video_path=video_path,
        pts_ticks=pts_ticks,
        relative_times_s=relative_times_s,
        key_frames=key_frames,
        time_base_num=time_base_num,
        time_base_den=time_base_den,
        pts_origin_tick=origin,
        width=width,
        height=height,
    )


def _try_probe_h264_packet_index(
    *,
    video_path: Path,
    ffprobe_executable: str,
    timeout_s: float,
) -> VideoFrameIndex | None:
    """
    Use demuxed packets only for the narrow case where packet order is display order.

    Progressive H.264 without B-frames has one presentation-ordered access unit per
    ordinary video packet in the recordings supported here.  Any metadata, packet,
    or ordering anomaly rejects this optimization and lets the decoded-frame probe
    below remain the source of truth.
    """
    if video_path.suffix.lower() not in {".mkv", ".mp4", ".m4v", ".mov"}:
        return None
    command = [
        ffprobe_executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_packets",
        "-show_entries",
        (
            "stream=codec_name,time_base,width,height,field_order,has_b_frames,nb_frames:"
            "stream_disposition=attached_pic:packet=pts,dts,flags"
        ),
        "-of",
        "json",
        str(video_path),
    ]
    try:
        payload = _run_ffprobe_json(
            command, video_path=video_path, timeout_s=timeout_s
        )
        streams = payload.get("streams", [])
        if not isinstance(streams, list) or len(streams) != 1:
            return None
        stream = streams[0]
        if not isinstance(stream, dict):
            return None
        disposition = stream.get("disposition", {})
        if not isinstance(disposition, dict):
            return None
        if (
            str(stream.get("codec_name", "")).lower() != "h264"
            or str(stream.get("field_order", "")).lower() != "progressive"
            or int(stream.get("has_b_frames", -1)) != 0
            or int(disposition.get("attached_pic", -1)) != 0
        ):
            return None

        packets = payload.get("packets", [])
        if not isinstance(packets, list) or not packets:
            return None
        raw_frame_count = stream.get("nb_frames")
        if raw_frame_count not in (None, "", "N/A"):
            if int(raw_frame_count) != len(packets):
                return None
        pts_values: list[int] = []
        key_frame_values: list[bool] = []
        for packet in packets:
            if not isinstance(packet, dict):
                return None
            raw_pts = packet.get("pts")
            raw_dts = packet.get("dts")
            flags = str(packet.get("flags", ""))
            if raw_pts in (None, "", "N/A") or raw_dts in (None, "", "N/A"):
                return None
            pts = int(raw_pts)
            dts = int(raw_dts)
            # With no B-frames, a differing decode timestamp is an unexpected
            # reorder/delay signal; reject rather than infer display-frame order.
            if pts != dts or "D" in flags or "C" in flags:
                return None
            pts_values.append(pts)
            key_frame_values.append("K" in flags)

        # Reject missing/late random-access metadata.  The slower frame probe
        # can still index unusual but valid streams safely.
        if not key_frame_values[0] or not any(key_frame_values):
            return None
        return _make_video_frame_index(
            video_path=video_path,
            stream=stream,
            pts_values=pts_values,
            key_frame_values=key_frame_values,
        )
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError, OverflowError):
        return None


@lru_cache(maxsize=128)
def _probe_video_frame_index_cached(
    video_path_string: str,
    ffprobe_executable: str,
    timeout_s: float,
) -> VideoFrameIndex:
    video_path = Path(video_path_string)
    packet_index = _try_probe_h264_packet_index(
        video_path=video_path,
        ffprobe_executable=ffprobe_executable,
        timeout_s=timeout_s,
    )
    if packet_index is not None:
        return packet_index

    command = [
        ffprobe_executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_frames",
        "-show_entries",
        "stream=time_base,width,height:frame=best_effort_timestamp,key_frame",
        "-of",
        "json",
        str(video_path),
    ]
    payload = _run_ffprobe_json(
        command, video_path=video_path, timeout_s=timeout_s
    )

    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(
            f"Expected exactly one selected video stream in {video_path}, got {len(streams)}"
        )
    stream = streams[0]
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError(f"No displayed video frames were found in {video_path}")

    pts_values: list[int] = []
    key_frame_values: list[bool] = []
    for frame_number, frame in enumerate(frames):
        raw_tick = frame.get("best_effort_timestamp")
        if raw_tick in (None, "", "N/A"):
            raise ValueError(
                f"Frame {frame_number} in {video_path} has no valid presentation timestamp"
            )
        try:
            pts_values.append(int(raw_tick))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Frame {frame_number} in {video_path} has invalid PTS {raw_tick!r}"
            ) from error
        key_frame_values.append(str(frame.get("key_frame", "0")) == "1")

    return _make_video_frame_index(
        video_path=video_path,
        stream=stream,
        pts_values=pts_values,
        key_frame_values=key_frame_values,
    )


def probe_video_frame_index(
    video_path: str | Path,
    *,
    ffprobe_path: str | Path | None = None,
    timeout_s: float = 300.0,
) -> VideoFrameIndex:
    """Build a strict presentation-order frame index from integer PTS values."""
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    timeout_value = float(timeout_s)
    if not math.isfinite(timeout_value) or timeout_value <= 0:
        raise ValueError("timeout_s must be finite and positive")
    frame_index = _probe_video_frame_index_cached(
        str(path),
        _resolve_ffprobe(ffprobe_path),
        timeout_value,
    )
    return frame_index


def nearest_frame_number(frame_index: VideoFrameIndex, relative_time_s: float) -> int:
    """Return the displayed-frame number nearest to a relative video time."""
    time_s = float(relative_time_s)
    if not math.isfinite(time_s):
        raise ValueError("relative_time_s must be finite")
    times = frame_index.relative_times_s
    insertion = int(np.searchsorted(times, time_s, side="left"))
    if insertion <= 0:
        return 0
    if insertion >= frame_index.frame_count:
        return frame_index.frame_count - 1
    before = insertion - 1
    return before if abs(times[before] - time_s) <= abs(times[insertion] - time_s) else insertion


def _exact_frame_number(value: object, *, name: str = "frame_number") -> int:
    """Accept an integer value without silently truncating fractional input."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be a finite integer")
    return int(numeric)


class VerifiedFrameReader:
    """
    Decode and verify every requested frame against the ffprobe PTS index.

    OpenCV handles sequential reads and containers whose frame-number seeking
    is trustworthy.  Some MKV backends report that a random seek succeeded but
    silently return frame zero; those requests fall back to an exact FFmpeg PTS
    seek whose emitted PTS is checked against the independent ffprobe index.
    There is no FPS-derived fallback.
    """

    def __init__(
        self,
        frame_index: VideoFrameIndex,
        *,
        cache_size: int = 8,
        ffmpeg_path: str | Path | None = None,
        ffmpeg_timeout_s: float = 60.0,
    ) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        timeout_value = float(ffmpeg_timeout_s)
        if not math.isfinite(timeout_value) or timeout_value <= 0:
            raise ValueError("ffmpeg_timeout_s must be finite and positive")
        self.frame_index = frame_index
        self.cache_size = int(cache_size)
        self.ffmpeg_path = ffmpeg_path
        self.ffmpeg_timeout_s = timeout_value
        self._ffmpeg_executable: str | None = None
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._capture: cv2.VideoCapture | None = None
        self._origin_position_ms: float | None = None
        self._last_decoded_frame: int | None = None
        self._opencv_random_seek_supported = True
        self._open()

    def __enter__(self) -> "VerifiedFrameReader":
        return self

    def __exit__(self, _exception_type, _exception, _traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._cache.clear()
        self._last_decoded_frame = None

    def _open(self) -> None:
        capture = cv2.VideoCapture(str(self.frame_index.video_path), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(str(self.frame_index.video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video: {self.frame_index.video_path}")
        self._capture = capture

        reported_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._opencv_random_seek_supported = bool(
            math.isfinite(reported_count)
            and reported_count > 0
            and int(round(reported_count)) == self.frame_index.frame_count
        )

        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            self.close()
            raise RuntimeError(
                f"Could not decode the first frame from {self.frame_index.video_path}"
            )
        position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        if not math.isfinite(position_ms):
            self.close()
            raise RuntimeError(
                "The decoder did not report a finite presentation time for frame 0"
            )
        self._origin_position_ms = position_ms
        self._verify_decoded_frame(0, frame)
        self._remember(0, frame)
        self._last_decoded_frame = 0

    def _time_tolerance_s(self, frame_number: int) -> float:
        times = self.frame_index.relative_times_s
        intervals: list[float] = []
        if frame_number > 0:
            intervals.append(float(times[frame_number] - times[frame_number - 1]))
        if frame_number + 1 < self.frame_index.frame_count:
            intervals.append(float(times[frame_number + 1] - times[frame_number]))
        local_interval = min(intervals) if intervals else 0.0
        return max(0.001, local_interval * 0.1)

    def _verify_decoded_frame(self, expected_number: int, frame: np.ndarray) -> None:
        if self._capture is None or self._origin_position_ms is None:
            raise RuntimeError("Frame reader is closed")
        if frame.shape[1] != self.frame_index.width or frame.shape[0] != self.frame_index.height:
            raise RuntimeError(
                "Decoded frame dimensions do not match the indexed video stream: "
                f"{frame.shape[1]}x{frame.shape[0]} vs "
                f"{self.frame_index.width}x{self.frame_index.height}."
            )

        reported_next = float(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        if not math.isfinite(reported_next) or reported_next <= 0:
            raise RuntimeError(
                "The decoder cannot report its decoded frame number; exact annotation "
                "cannot be verified."
            )
        reported_number = int(round(reported_next)) - 1
        if reported_number != expected_number:
            raise RuntimeError(
                f"Decoder returned frame {reported_number}, expected {expected_number}, "
                f"in {self.frame_index.video_path}."
            )

        reported_position_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
        if not math.isfinite(reported_position_ms):
            raise RuntimeError(
                f"Decoder returned no finite PTS for frame {expected_number}"
            )
        actual_relative_s = (reported_position_ms - self._origin_position_ms) / 1000.0
        expected_relative_s = float(
            self.frame_index.relative_times_s[expected_number]
        )
        tolerance = self._time_tolerance_s(expected_number)
        if not math.isclose(
            actual_relative_s,
            expected_relative_s,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise RuntimeError(
                "Decoded-frame PTS does not match the ffprobe index: "
                f"frame {expected_number}, decoder={actual_relative_s:.9f}s, "
                f"ffprobe={expected_relative_s:.9f}s, tolerance={tolerance:.9f}s."
            )

    def _remember(self, frame_number: int, frame: np.ndarray) -> None:
        self._cache[frame_number] = frame.copy()
        self._cache.move_to_end(frame_number)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _decode_next(self, expected_number: int) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("Frame reader is closed")
        ok, frame = self._capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError(
                f"Could not decode frame {expected_number} from "
                f"{self.frame_index.video_path}"
            )
        self._verify_decoded_frame(expected_number, frame)
        self._remember(expected_number, frame)
        self._last_decoded_frame = expected_number
        return frame

    def _preceding_keyframe(self, frame_number: int) -> int:
        candidates = np.flatnonzero(self.frame_index.key_frames[: frame_number + 1])
        return int(candidates[-1]) if candidates.size else 0

    def _read_exact_with_ffmpeg(self, frame_number: int) -> np.ndarray:
        """Decode one frame by absolute PTS and verify the PTS FFmpeg emitted."""
        if self._ffmpeg_executable is None:
            self._ffmpeg_executable = _resolve_ffmpeg(self.ffmpeg_path)

        target_tick = int(self.frame_index.pts_ticks[frame_number])
        keyframe_number = self._preceding_keyframe(frame_number)
        keyframe_tick = int(self.frame_index.pts_ticks[keyframe_number])
        keyframe_seconds = (
            keyframe_tick
            * self.frame_index.time_base_num
            / self.frame_index.time_base_den
        )
        command = [
            self._ffmpeg_executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-copyts",
            "-seek_timestamp",
            "1",
            "-ss",
            f"{keyframe_seconds:.12f}",
            "-i",
            str(self.frame_index.video_path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-frames:v",
            "1",
            "-vf",
            f"select=eq(pts\\,{target_tick}),showinfo",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        run_kwargs: dict[str, object] = {
            "capture_output": True,
            "check": False,
            "timeout": self.ffmpeg_timeout_s,
        }
        if os.name == "nt":
            run_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            completed = subprocess.run(command, **run_kwargs)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                "FFmpeg did not finish the exact seek to "
                f"frame {frame_number} within {self.ffmpeg_timeout_s:g} s"
            ) from error

        stderr = completed.stderr.decode("utf-8", errors="replace")
        if completed.returncode != 0:
            detail = stderr.strip() or "unknown FFmpeg error"
            raise RuntimeError(
                f"FFmpeg could not decode frame {frame_number} from "
                f"{self.frame_index.video_path}: {detail}"
            )

        emitted_ticks = [
            int(value)
            for value in re.findall(r"\bpts:\s*(-?\d+)\s+pts_time:", stderr)
        ]
        if emitted_ticks != [target_tick]:
            raise RuntimeError(
                "FFmpeg exact seek did not emit the requested PTS: "
                f"frame {frame_number}, expected tick {target_tick}, "
                f"reported {emitted_ticks or 'none'}."
            )

        expected_bytes = self.frame_index.width * self.frame_index.height * 3
        if len(completed.stdout) != expected_bytes:
            raise RuntimeError(
                "FFmpeg returned an incomplete decoded frame: "
                f"expected {expected_bytes:,} bytes, got {len(completed.stdout):,}."
            )
        frame = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
            self.frame_index.height,
            self.frame_index.width,
            3,
        )
        self._remember(frame_number, frame)
        return frame.copy()

    def read(self, frame_number: int) -> np.ndarray:
        number = _exact_frame_number(frame_number)
        if number < 0 or number >= self.frame_index.frame_count:
            raise IndexError(
                f"Frame {number} is outside [0, {self.frame_index.frame_count - 1}]"
            )
        if number in self._cache:
            frame = self._cache[number]
            self._cache.move_to_end(number)
            return frame.copy()
        if self._capture is None:
            raise RuntimeError("Frame reader is closed")

        start_number: int
        used_random_seek = False
        if (
            self._last_decoded_frame is not None
            and self._last_decoded_frame < number
            and number - self._last_decoded_frame <= 300
        ):
            start_number = self._last_decoded_frame + 1
        else:
            if not self._opencv_random_seek_supported:
                return self._read_exact_with_ffmpeg(number)
            used_random_seek = True
            keyframe = self._preceding_keyframe(number)
            seek_ok = self._capture.set(cv2.CAP_PROP_POS_FRAMES, keyframe)
            if not seek_ok:
                self._opencv_random_seek_supported = False
                self._last_decoded_frame = None
                return self._read_exact_with_ffmpeg(number)
            start_number = keyframe

        frame: np.ndarray | None = None
        try:
            for expected_number in range(start_number, number + 1):
                frame = self._decode_next(expected_number)
        except RuntimeError as opencv_error:
            # A number of MKV/OpenCV combinations return True from set() but
            # restart decoding at frame zero.  Do not trust random seeks from
            # that capture again; verify an exact PTS decode with FFmpeg.
            if not used_random_seek:
                raise
            self._opencv_random_seek_supported = False
            self._last_decoded_frame = None
            try:
                return self._read_exact_with_ffmpeg(number)
            except Exception as ffmpeg_error:
                raise RuntimeError(
                    f"OpenCV random seek failed ({opencv_error}); FFmpeg exact "
                    f"seek also failed ({ffmpeg_error})."
                ) from ffmpeg_error
        if frame is None:
            raise RuntimeError(f"Could not reach frame {number}")
        return frame.copy()


def _validate_roi(
    roi_norm: Sequence[float] | None,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    if roi_norm is None:
        return None
    if len(roi_norm) != 4:
        raise ValueError("roi_norm must contain (left, top, right, bottom)")
    left, top, right, bottom = (float(value) for value in roi_norm)
    values = (left, top, right, bottom)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("ROI values must be finite")
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError(
            "ROI must satisfy 0 <= left < right <= 1 and "
            "0 <= top < bottom <= 1"
        )
    left_px, top_px, right_px, bottom_px = _roi_pixels(values, width, height)
    if right_px - left_px < 2 or bottom_px - top_px < 2:
        raise ValueError("ROI maps to fewer than 2 pixels in one dimension")
    return values


def _roi_pixels(
    roi_norm: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = roi_norm
    return (
        max(0, min(width - 1, int(round(left * width)))),
        max(0, min(height - 1, int(round(top * height)))),
        max(1, min(width, int(round(right * width)))),
        max(1, min(height, int(round(bottom * height)))),
    )


def _draw_roi(frame_bgr: np.ndarray, roi_norm: Sequence[float] | None) -> np.ndarray:
    rendered = frame_bgr.copy()
    if roi_norm is not None:
        height, width = rendered.shape[:2]
        left, top, right, bottom = _roi_pixels(roi_norm, width, height)
        cv2.rectangle(rendered, (left, top), (right - 1, bottom - 1), (0, 255, 255), 3)
    return rendered


def _format_time(seconds: float) -> str:
    minutes, remainder = divmod(float(seconds), 60.0)
    return f"{int(minutes):02d}:{remainder:06.3f}"


def show_transition_bracket(
    frame_index: VideoFrameIndex,
    *,
    last_old_frame: int,
    first_new_frame: int,
    event: str,
    roi_norm: Sequence[float] | None = None,
):
    """Show the two selected transition bounds side by side for final review."""
    import matplotlib.pyplot as plt

    if event not in EVENTS:
        raise ValueError(f"event must be one of {EVENTS}")
    old_number = _exact_frame_number(last_old_frame, name="last_old_frame")
    new_number = _exact_frame_number(first_new_frame, name="first_new_frame")
    if not 0 <= old_number < new_number < frame_index.frame_count:
        raise ValueError("Require 0 <= last_old_frame < first_new_frame < frame_count")
    roi = _validate_roi(roi_norm, width=frame_index.width, height=frame_index.height)
    states = ("ON", "OFF") if event == "panel_off" else ("OFF", "ON")
    numbers = (old_number, new_number)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    with VerifiedFrameReader(frame_index, cache_size=2) as reader:
        for axis, number, state, label in zip(
            axes, numbers, states, ("last clear old state", "first clear new state")
        ):
            frame = reader.read(number)
            rendered = cv2.cvtColor(_draw_roi(frame, roi), cv2.COLOR_BGR2RGB)
            axis.imshow(rendered)
            axis.set_title(
                f"{label}: {state}\nframe {number} | "
                f"{_format_time(frame_index.relative_times_s[number])}"
            )
            axis.axis("off")
    width_s = float(
        frame_index.relative_times_s[numbers[1]]
        - frame_index.relative_times_s[numbers[0]]
    )
    figure.suptitle(
        f"{event}: bracket {width_s:.6f} s, uncertainty ±{width_s / 2:.6f} s"
    )
    figure.tight_layout()
    return figure


def _empty_annotations() -> pd.DataFrame:
    return pd.DataFrame(columns=ANNOTATION_COLUMNS)


def _read_annotation_file(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        return _empty_annotations()
    # Read text literally: a note such as "NA" or "null" is user content, not
    # a pandas missing-value token. Numeric validation is explicit downstream.
    frame = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    missing = [column for column in ANNOTATION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Annotation file {csv_path} is missing columns: {missing}"
        )
    return frame.loc[:, ANNOTATION_COLUMNS].copy()


def load_panel_annotations(
    csv_path: str | Path,
) -> pd.DataFrame:
    """Load the single current annotation CSV."""
    return _read_annotation_file(Path(csv_path))


_INTEGER_COLUMNS = {
    "pts_time_base_num",
    "pts_time_base_den",
    "pts_origin_tick",
    "last_old_pts_tick",
    "first_new_pts_tick",
}


def _is_missing_cell(value: object) -> bool:
    return value is None or (not isinstance(value, str) and pd.isna(value)) or value == ""


def _integer_cell(value: object, column: str) -> str:
    if _is_missing_cell(value):
        return ""
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    text = str(value).strip()
    if text.startswith(("+", "-")):
        digits = text[1:]
    else:
        digits = text
    if digits.isdigit():
        return str(int(text))
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column} must be an integer") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{column} must be an integer")
    return str(int(numeric))


def _serializable_annotations(frame: pd.DataFrame) -> pd.DataFrame:
    serializable = frame.loc[:, ANNOTATION_COLUMNS].copy()
    for column in ANNOTATION_COLUMNS:
        if column in _INTEGER_COLUMNS:
            serializable[column] = serializable[column].map(
                lambda value, name=column: _integer_cell(value, name)
            )
        else:
            serializable[column] = serializable[column].map(
                lambda value: "" if _is_missing_cell(value) else str(value)
            )
    return serializable


def annotation_bracket_values(
    annotation: Mapping[str, object] | pd.Series,
) -> tuple[float, float, float, float]:
    """Return relative midpoint, raw midpoint, uncertainty, and PTS origin."""
    numerator = int(_integer_cell(annotation["pts_time_base_num"], "pts_time_base_num"))
    denominator = int(
        _integer_cell(annotation["pts_time_base_den"], "pts_time_base_den")
    )
    origin = int(_integer_cell(annotation["pts_origin_tick"], "pts_origin_tick"))
    old_tick = int(
        _integer_cell(annotation["last_old_pts_tick"], "last_old_pts_tick")
    )
    new_tick = int(
        _integer_cell(annotation["first_new_pts_tick"], "first_new_pts_tick")
    )
    if numerator <= 0 or denominator <= 0 or old_tick >= new_tick:
        raise ValueError("invalid PTS transition bracket")
    old_relative = float(old_tick - origin) * numerator / denominator
    new_relative = float(new_tick - origin) * numerator / denominator
    relative_midpoint = (old_relative + new_relative) / 2
    uncertainty = (new_relative - old_relative) / 2
    origin_s = origin * numerator / denominator
    return relative_midpoint, relative_midpoint + origin_s, uncertainty, origin_s


def _write_annotation_file(frame: pd.DataFrame, csv_path: Path) -> None:
    """Write the one current row per panel key to the configured CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _serializable_annotations(frame).to_csv(csv_path, index=False)


def _annotation_mask(frame: pd.DataFrame, values: Mapping[str, str]) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for column, value in values.items():
        mask &= frame[column].astype(str) == str(value)
    return mask


def save_panel_annotation(
    *,
    annotation_csv: str | Path,
    project: str,
    pid: str,
    visit: str,
    source: str,
    event: str,
    frame_index: VideoFrameIndex,
    status: str,
    last_old_frame: int | None = None,
    first_new_frame: int | None = None,
    confidence: str = "not_rated",
    notes: str = "",
    segment_id: str = "main",
) -> pd.Series:
    """Save one decision and its exact PTS bracket."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if event not in EVENTS:
        raise ValueError(f"event must be one of {EVENTS}")
    if status not in ANNOTATION_STATUSES:
        raise ValueError(f"status must be one of {ANNOTATION_STATUSES}")

    has_bounds = last_old_frame is not None or first_new_frame is not None
    if status == "accepted" and (last_old_frame is None or first_new_frame is None):
        raise ValueError("An accepted annotation requires both transition-bound frames")
    if has_bounds and (last_old_frame is None or first_new_frame is None):
        raise ValueError("Provide both transition-bound frames or neither")

    old_tick: object = np.nan
    new_tick: object = np.nan
    if has_bounds:
        old_number = _exact_frame_number(last_old_frame, name="last_old_frame")
        new_number = _exact_frame_number(first_new_frame, name="first_new_frame")
        if not 0 <= old_number < new_number < frame_index.frame_count:
            raise ValueError(
                "Require 0 <= last_old_frame < first_new_frame < frame_count"
            )
        old_tick = int(frame_index.pts_ticks[old_number])
        new_tick = int(frame_index.pts_ticks[new_number])

    annotation_path = Path(annotation_csv)
    key = {
        "project": str(project).strip(),
        "pid": str(pid).strip(),
        "visit": str(visit).strip(),
        "segment_id": str(segment_id).strip(),
        "source": source,
        "event": event,
    }
    current = _read_annotation_file(annotation_path)

    row: dict[str, object] = {
        **key,
        "status": status,
        "confidence": confidence,
        "notes": "" if _is_missing_cell(notes) else str(notes),
        "pts_time_base_num": frame_index.time_base_num,
        "pts_time_base_den": frame_index.time_base_den,
        "pts_origin_tick": frame_index.pts_origin_tick,
        "last_old_pts_tick": old_tick,
        "first_new_pts_tick": new_tick,
    }
    new_row = pd.DataFrame([row], columns=ANNOTATION_COLUMNS)
    retained = current.loc[
        ~_annotation_mask(current, key), ANNOTATION_COLUMNS
    ]
    updated = (
        new_row.copy()
        if retained.empty
        else pd.concat([retained, new_row], ignore_index=True)
    ).sort_values(ANNOTATION_KEY_COLUMNS, kind="stable", ignore_index=True)
    _write_annotation_file(updated, annotation_path)
    return new_row.iloc[0].copy()


def summarize_panel_annotations(
    video_pairs: pd.DataFrame,
    annotation_csv: str | Path,
    *,
    segment_id: str = "main",
    annotations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize the four required manual events for every discovered visit."""
    current = (
        load_panel_annotations(annotation_csv)
        if annotations is None
        else annotations.copy()
    )
    rows: list[dict[str, object]] = []
    for _, pair in video_pairs.iterrows():
        result: dict[str, object] = {
            "project": pair["project"],
            "pid": pair["pid"],
            "visit": pair["visit"],
            "segment_id": str(segment_id),
            "discovery_status": pair["discovery_status"],
        }
        all_accepted = True
        accepted_times: dict[str, float] = {}
        for source, path_column in (("driver", "driver_path"), ("et_scene", "et_scene_path")):
            expected_path = pair[path_column]
            for event in EVENTS:
                label = f"{source}_{event}"
                key = {
                    "project": str(pair["project"]),
                    "pid": str(pair["pid"]),
                    "visit": str(pair["visit"]),
                    "segment_id": str(segment_id),
                    "source": source,
                    "event": event,
                }
                selected = current.loc[_annotation_mask(current, key)]
                if pd.isna(expected_path):
                    state = "video_missing_or_ambiguous"
                elif selected.empty:
                    state = "missing"
                else:
                    row = selected.iloc[-1]
                    state = str(row["status"])
                    if state == "accepted":
                        try:
                            relative, raw, _, _ = annotation_bracket_values(row)
                            accepted_times[label] = (
                                raw if source == "et_scene" else relative
                            )
                        except (TypeError, ValueError):
                            state = "invalid_bracket"
                result[label] = state
                all_accepted &= state == "accepted"
        order_valid = True
        for source in SOURCES:
            off_key = f"{source}_panel_off"
            on_key = f"{source}_panel_on"
            if off_key in accepted_times and on_key in accepted_times:
                order_valid &= accepted_times[off_key] < accepted_times[on_key]
        result["event_order_valid"] = order_valid
        result["anchors_complete"] = (
            pair["discovery_status"] == "ready" and all_accepted and order_valid
        )
        rows.append(result)
    return pd.DataFrame(rows)


def _jpeg_bytes(
    frame_bgr: np.ndarray,
    roi_norm: Sequence[float] | None,
    *,
    max_width: int = 1000,
) -> bytes:
    rendered = _draw_roi(frame_bgr, roi_norm)
    if rendered.shape[1] > max_width:
        scale = max_width / rendered.shape[1]
        rendered = cv2.resize(
            rendered,
            (max_width, max(1, int(round(rendered.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Could not encode the annotation preview frame")
    return encoded.tobytes()


def _roi_crop_jpeg_bytes(
    frame_bgr: np.ndarray,
    roi_norm: Sequence[float],
    *,
    max_width: int = 700,
) -> bytes:
    height, width = frame_bgr.shape[:2]
    left, top, right, bottom = _roi_pixels(roi_norm, width, height)
    return _jpeg_bytes(
        frame_bgr[top:bottom, left:right], None, max_width=max_width
    )


class ManualPanelAnnotationTool:
    """Optional ipywidgets front end for the strict function-based workflow."""

    def __init__(
        self,
        video_pairs: pd.DataFrame,
        *,
        annotation_csv: str | Path,
        ffprobe_path: str | Path | None = None,
        ffmpeg_path: str | Path | None = None,
        segment_id: str = "main",
        ffprobe_timeout_s: float = 300.0,
        ffmpeg_timeout_s: float = 60.0,
    ) -> None:
        try:
            import ipywidgets as widgets
        except ImportError as error:
            raise ImportError(
                "The optional widget needs ipywidgets. Use probe_video_frame_index(), "
                "show_transition_bracket(), and "
                "save_panel_annotation() instead."
            ) from error

        if video_pairs.empty:
            raise ValueError("video_pairs is empty; review DATA_ROOT and discovery output")
        if not set(PAIR_COLUMNS).issubset(video_pairs.columns):
            raise ValueError("video_pairs does not have the discovery schema")
        self._widgets = widgets
        self.video_pairs = video_pairs.reset_index(drop=True).copy()
        self.annotation_csv = Path(annotation_csv)
        self.ffprobe_path = ffprobe_path
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_timeout_s = float(ffprobe_timeout_s)
        self.ffmpeg_timeout_s = float(ffmpeg_timeout_s)
        self.segment_id = str(segment_id)
        self.frame_index: VideoFrameIndex | None = None
        self._reader: VerifiedFrameReader | None = None
        self.current_frame = 0
        self.last_old_frame: int | None = None
        self.first_new_frame: int | None = None

        pair_options = [
            (
                f"{row.pid} {row.visit} [{row.discovery_status}]",
                int(index),
            )
            for index, row in self.video_pairs.iterrows()
        ]
        self.pair = widgets.Dropdown(options=pair_options, description="Visit:")
        self.source = widgets.Dropdown(
            options=[("Driver video", "driver"), ("ET scene video", "et_scene")],
            description="Source:",
        )
        self.event = widgets.Dropdown(
            options=[("Panel off (ON → OFF)", "panel_off"), ("Panel on (OFF → ON)", "panel_on")],
            description="Event:",
        )
        self.coarse_time = widgets.FloatText(value=0.0, description="Coarse s:")
        self.load_button = widgets.Button(
            description="Load video / record", button_style="info"
        )
        self.jump_button = widgets.Button(description="Jump to coarse s")

        self.back_ten_minutes = widgets.Button(description="−10 min")
        self.back_minute = widgets.Button(description="−1 min")
        self.back_ten_seconds = widgets.Button(description="−10 s")
        self.back_second = widgets.Button(description="−1 s")
        self.back_ten = widgets.Button(description="−10 frames")
        self.back_one = widgets.Button(description="−1 frame")
        self.forward_one = widgets.Button(description="+1 frame")
        self.forward_ten = widgets.Button(description="+10 frames")
        self.forward_second = widgets.Button(description="+1 s")
        self.forward_ten_seconds = widgets.Button(description="+10 s")
        self.forward_minute = widgets.Button(description="+1 min")
        self.forward_ten_minutes = widgets.Button(description="+10 min")

        self.mark_old = widgets.Button(description="Mark last ON", button_style="warning")
        self.mark_new = widgets.Button(description="Mark first OFF", button_style="success")
        self.clear_marks = widgets.Button(description="Clear marks")
        self.review_button = widgets.Button(
            description="Review bracket", button_style="info"
        )
        # A bytes-backed widgets.Image renders an empty ``src`` as a broken
        # image in some Notebook/JupyterLab front ends, and older widget
        # managers can also mishandle its binary comm buffer. Output widgets
        # carrying the standard ``image/jpeg`` MIME bundle are more portable
        # and remain genuinely blank until a frame has been loaded.
        preview_layout = widgets.Layout(width="100%", overflow="auto")
        self.preview = widgets.Output(layout=preview_layout)
        self.roi_preview = widgets.Output(layout=preview_layout)
        self.frame_label = widgets.HTML(
            value="No frame loaded. Click <b>Load video / record</b>."
        )
        self.mark_label = widgets.HTML()

        roi_style = {"description_width": "70px"}
        self.roi_left = widgets.Text(value="", description="ROI left:", style=roi_style)
        self.roi_top = widgets.Text(value="", description="top:", style=roi_style)
        self.roi_right = widgets.Text(value="", description="right:", style=roi_style)
        self.roi_bottom = widgets.Text(value="", description="bottom:", style=roi_style)
        self.apply_roi = widgets.Button(description="Apply ROI")

        self.status = widgets.Dropdown(
            options=ANNOTATION_STATUSES,
            value="draft",
            description="Status:",
        )
        self.confidence = widgets.Dropdown(
            options=CONFIDENCE_LEVELS,
            value="not_rated",
            description="Confidence:",
        )
        self.notes = widgets.Textarea(value="", description="Notes:")
        self.save_button = widgets.Button(description="Save annotation", button_style="primary")
        self.output = widgets.Output()

        self.widget = widgets.VBox(
            [
                widgets.HTML(
                    "<b>Manual panel transition annotation</b><br>"
                    "Times are relative to the first displayed video frame and come from integer PTS. "
                    "Yellow rectangle = optional normalized ROI."
                ),
                widgets.HBox([self.pair, self.source, self.event]),
                widgets.HBox([self.coarse_time, self.load_button, self.jump_button]),
                self.output,
                self.frame_label,
                widgets.HTML("<b>Frame preview</b>"),
                self.preview,
                widgets.HTML(
                    "<b>ROI crop (original brightness; blank when no ROI is set)</b>"
                ),
                self.roi_preview,
                widgets.HBox(
                    [
                        self.back_ten_minutes,
                        self.back_minute,
                        self.back_ten_seconds,
                        self.back_second,
                        self.forward_second,
                        self.forward_ten_seconds,
                        self.forward_minute,
                        self.forward_ten_minutes,
                    ]
                ),
                widgets.HBox(
                    [self.back_ten, self.back_one, self.forward_one, self.forward_ten]
                ),
                widgets.HBox(
                    [self.mark_old, self.mark_new, self.clear_marks, self.review_button]
                ),
                self.mark_label,
                widgets.HBox([self.roi_left, self.roi_top, self.roi_right, self.roi_bottom, self.apply_roi]),
                widgets.HBox([self.status, self.confidence]),
                self.notes,
                self.save_button,
            ]
        )

        self.load_button.on_click(self._on_load)
        self.jump_button.on_click(self._on_jump)
        self.back_one.on_click(lambda _: self._move_frames(-1))
        self.back_ten.on_click(lambda _: self._move_frames(-10))
        self.forward_one.on_click(lambda _: self._move_frames(1))
        self.forward_ten.on_click(lambda _: self._move_frames(10))
        self.back_ten_minutes.on_click(lambda _: self._move_seconds(-600.0))
        self.back_minute.on_click(lambda _: self._move_seconds(-60.0))
        self.back_ten_seconds.on_click(lambda _: self._move_seconds(-10.0))
        self.back_second.on_click(lambda _: self._move_seconds(-1.0))
        self.forward_second.on_click(lambda _: self._move_seconds(1.0))
        self.forward_ten_seconds.on_click(lambda _: self._move_seconds(10.0))
        self.forward_minute.on_click(lambda _: self._move_seconds(60.0))
        self.forward_ten_minutes.on_click(lambda _: self._move_seconds(600.0))
        self.mark_old.on_click(self._on_mark_old)
        self.mark_new.on_click(self._on_mark_new)
        self.clear_marks.on_click(self._on_clear_marks)
        self.review_button.on_click(self._on_review)
        self.apply_roi.on_click(self._on_apply_roi)
        self.save_button.on_click(self._on_save)
        self.event.observe(self._on_event_changed, names="value")
        self.pair.observe(self._on_selection_changed, names="value")
        self.source.observe(self._on_selection_changed, names="value")
        self._update_mark_button_labels()
        self._update_mark_label()

    def display(self):
        """Display the widget in a Jupyter notebook and return this tool."""
        from IPython.display import display

        display(self.widget)
        return self

    def close(self) -> None:
        """Release the decoder held for responsive frame stepping."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        # ``__del__`` can run after a partially constructed widget, hence the
        # guarded lookup rather than assuming both outputs always exist.
        for name in ("preview", "roi_preview"):
            output_widget = getattr(self, name, None)
            if output_widget is not None:
                output_widget.outputs = ()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _selected_pair(self) -> pd.Series:
        return self.video_pairs.iloc[int(self.pair.value)]

    def _selected_video_path(self) -> Path:
        row = self._selected_pair()
        column = "driver_path" if self.source.value == "driver" else "et_scene_path"
        value = row[column]
        if pd.isna(value):
            candidate_column = "driver_candidates" if self.source.value == "driver" else "et_scene_candidates"
            raise ValueError(
                f"This source is missing or ambiguous. Candidates: {row[candidate_column] or '(none)'}"
            )
        return Path(str(value))

    def _write_message(self, message: str) -> None:
        with self.output:
            self.output.clear_output(wait=True)
            print(message)

    def _clear_previews(self) -> None:
        """Clear frame outputs without leaving an empty broken-image element."""
        self.preview.outputs = ()
        self.roi_preview.outputs = ()

    @staticmethod
    def _display_jpeg(output_widget, jpeg_bytes: bytes) -> None:
        """Render JPEG bytes through Jupyter's standard image MIME display."""
        from IPython.display import Image as IPythonImage

        output_widget.outputs = ()
        output_widget.append_display_data(
            IPythonImage(data=jpeg_bytes, format="jpeg", embed=True)
        )

    def _run(self, action) -> None:
        try:
            action()
        except Exception as error:
            self._write_message(f"{type(error).__name__}: {error}")

    def _on_selection_changed(self, _change) -> None:
        self.close()
        self.frame_index = None
        self._reset_form()
        self._clear_previews()
        self.frame_label.value = "Selection changed; click <b>Load video / record</b>."

    def _on_event_changed(self, _change) -> None:
        self.close()
        self.frame_index = None
        self._reset_form()
        self._clear_previews()
        self.frame_label.value = "Event changed; click <b>Load video / record</b>."
        self._update_mark_button_labels()

    def _reset_form(self) -> None:
        """Clear record-specific state so it cannot leak into another key."""
        self.last_old_frame = None
        self.first_new_frame = None
        self.current_frame = 0
        self.coarse_time.value = 0.0
        for widget in (self.roi_left, self.roi_top, self.roi_right, self.roi_bottom):
            widget.value = ""
        self.status.value = "draft"
        self.confidence.value = "not_rated"
        self.notes.value = ""
        self._update_mark_label()

    def _update_mark_button_labels(self) -> None:
        if self.event.value == "panel_off":
            self.mark_old.description = "Mark last ON"
            self.mark_new.description = "Mark first OFF"
        else:
            self.mark_old.description = "Mark last OFF"
            self.mark_new.description = "Mark first ON"

    def _roi_value(self) -> tuple[float, float, float, float] | None:
        texts = [
            self.roi_left.value.strip(),
            self.roi_top.value.strip(),
            self.roi_right.value.strip(),
            self.roi_bottom.value.strip(),
        ]
        if not any(texts):
            return None
        if not all(texts):
            raise ValueError("Fill all four ROI values or leave all four blank")
        if self.frame_index is None:
            raise RuntimeError("Load a video before applying an ROI")
        return _validate_roi(
            tuple(float(text) for text in texts),
            width=self.frame_index.width,
            height=self.frame_index.height,
        )

    def _load_existing(self) -> bool:
        annotations = load_panel_annotations(self.annotation_csv)
        pair = self._selected_pair()
        key = {
            "project": str(pair["project"]),
            "pid": str(pair["pid"]),
            "visit": str(pair["visit"]),
            "segment_id": self.segment_id,
            "source": self.source.value,
            "event": self.event.value,
        }
        existing = annotations.loc[_annotation_mask(annotations, key)]
        if existing.empty:
            return False
        row = existing.iloc[-1]
        self.status.value = str(row["status"])
        self.confidence.value = str(row["confidence"])
        self.notes.value = str(row["notes"])

        def frame_number(column: str) -> int | None:
            if _is_missing_cell(row[column]):
                return None
            tick = int(_integer_cell(row[column], column))
            matches = np.flatnonzero(self.frame_index.pts_ticks == tick)
            if matches.size != 1:
                raise ValueError(f"Saved {column} is not present in the selected video")
            return int(matches[0])

        self.last_old_frame = frame_number("last_old_pts_tick")
        self.first_new_frame = frame_number("first_new_pts_tick")
        if self.last_old_frame is not None and self.first_new_frame is not None:
            midpoint = (
                self.frame_index.relative_times_s[self.last_old_frame]
                + self.frame_index.relative_times_s[self.first_new_frame]
            ) / 2
            self.coarse_time.value = float(midpoint)
            self.current_frame = nearest_frame_number(self.frame_index, float(midpoint))
        self._write_message(
            f"Loaded the current annotation ({row['status']}). Saving replaces "
            "this panel key in the current CSV."
        )
        return True

    def _on_load(self, _button) -> None:
        def action() -> None:
            self.close()
            try:
                self.frame_index = None
                self._reset_form()
                path = self._selected_video_path()
                self._write_message("Reading the video's presentation timestamps with ffprobe …")
                self.frame_index = probe_video_frame_index(
                    path,
                    ffprobe_path=self.ffprobe_path,
                    timeout_s=self.ffprobe_timeout_s,
                )
                self._reader = VerifiedFrameReader(
                    self.frame_index,
                    ffmpeg_path=self.ffmpeg_path,
                    ffmpeg_timeout_s=self.ffmpeg_timeout_s,
                )
                self.last_old_frame = None
                self.first_new_frame = None
                loaded = self._load_existing()
                if not loaded:
                    self.current_frame = nearest_frame_number(
                        self.frame_index, self.coarse_time.value
                    )
                self._render()
                self._update_mark_label()
                if not loaded:
                    self._write_message(
                        f"Loaded {self.frame_index.frame_count:,} strict-PTS frames. "
                        f"Last displayed PTS is {self.frame_index.duration_s:.3f} s. "
                        "Jump or step to the transition and mark both bounds."
                    )
            except Exception:
                self.close()
                self.frame_index = None
                raise

        self._run(action)

    def _on_jump(self, _button) -> None:
        def action() -> None:
            if self.frame_index is None or self._reader is None:
                raise RuntimeError("Load a video first")
            self.current_frame = nearest_frame_number(
                self.frame_index, float(self.coarse_time.value)
            )
            self._render()

        self._run(action)

    def _render(self) -> None:
        if self.frame_index is None or self._reader is None:
            raise RuntimeError("Load a video first")
        roi = self._roi_value()
        frame = self._reader.read(self.current_frame)
        self._display_jpeg(self.preview, _jpeg_bytes(frame, roi))
        if roi is None:
            self.roi_preview.outputs = ()
        else:
            self._display_jpeg(
                self.roi_preview, _roi_crop_jpeg_bytes(frame, roi)
            )
        relative_time = float(self.frame_index.relative_times_s[self.current_frame])
        self.frame_label.value = (
            f"<b>{self.frame_index.video_path.name}</b> — frame "
            f"<b>{self.current_frame}</b> / {self.frame_index.frame_count - 1}; "
            f"relative PTS <b>{_format_time(relative_time)}</b> ({relative_time:.6f} s); "
            f"video last PTS {_format_time(self.frame_index.duration_s)}"
        )

    def _move_frames(self, delta: int) -> None:
        def action() -> None:
            if self.frame_index is None:
                raise RuntimeError("Load a video first")
            self.current_frame = max(
                0, min(self.frame_index.frame_count - 1, self.current_frame + delta)
            )
            self._render()

        self._run(action)

    def _move_seconds(self, delta_s: float) -> None:
        def action() -> None:
            if self.frame_index is None:
                raise RuntimeError("Load a video first")
            target = float(self.frame_index.relative_times_s[self.current_frame]) + delta_s
            self.current_frame = nearest_frame_number(self.frame_index, target)
            self._render()

        self._run(action)

    def _on_mark_old(self, _button) -> None:
        if self.frame_index is None:
            self._write_message("Load a video first")
            return
        self.last_old_frame = self.current_frame
        self._update_mark_label()

    def _on_mark_new(self, _button) -> None:
        if self.frame_index is None:
            self._write_message("Load a video first")
            return
        self.first_new_frame = self.current_frame
        self._update_mark_label()

    def _on_clear_marks(self, _button) -> None:
        self.last_old_frame = None
        self.first_new_frame = None
        self._update_mark_label()

    def _update_mark_label(self) -> None:
        if self.frame_index is None:
            self.mark_label.value = "Bounds: not loaded"
            return
        pieces = []
        for label, number in (
            ("last old", self.last_old_frame),
            ("first new", self.first_new_frame),
        ):
            if number is None:
                pieces.append(f"{label}: —")
            else:
                pieces.append(
                    f"{label}: frame {number} "
                    f"({_format_time(self.frame_index.relative_times_s[number])})"
                )
        if (
            self.last_old_frame is not None
            and self.first_new_frame is not None
            and self.last_old_frame < self.first_new_frame
        ):
            width = float(
                self.frame_index.relative_times_s[self.first_new_frame]
                - self.frame_index.relative_times_s[self.last_old_frame]
            )
            pieces.append(f"uncertainty: ±{width / 2:.6f} s")
        self.mark_label.value = "<br>".join(pieces)

    def _on_apply_roi(self, _button) -> None:
        self._run(self._render)

    def _on_review(self, _button) -> None:
        def action() -> None:
            if self.frame_index is None:
                raise RuntimeError("Load a video first")
            figure = show_transition_bracket(
                self.frame_index,
                last_old_frame=self.last_old_frame,
                first_new_frame=self.first_new_frame,
                event=self.event.value,
                roi_norm=self._roi_value(),
            )
            from IPython.display import display
            import matplotlib.pyplot as plt

            with self.output:
                self.output.clear_output(wait=True)
                try:
                    display(figure)
                finally:
                    # The widget owns this transient preview.  Closing the
                    # pyplot manager after display prevents hundreds of manual
                    # reviews from retaining full-resolution frames in memory.
                    plt.close(figure)
        self._run(action)

    def _on_save(self, _button) -> None:
        def action() -> None:
            if self.frame_index is None:
                raise RuntimeError("Load a video first")
            pair = self._selected_pair()
            row = save_panel_annotation(
                annotation_csv=self.annotation_csv,
                project=str(pair["project"]),
                pid=str(pair["pid"]),
                visit=str(pair["visit"]),
                segment_id=self.segment_id,
                source=self.source.value,
                event=self.event.value,
                frame_index=self.frame_index,
                status=self.status.value,
                last_old_frame=self.last_old_frame,
                first_new_frame=self.first_new_frame,
                confidence=self.confidence.value,
                notes=self.notes.value,
            )
            self._write_message(
                f"Saved current annotation: {row['status']} — {self.annotation_csv}"
            )

        self._run(action)


def launch_panel_annotator(
    video_pairs: pd.DataFrame,
    *,
    annotation_csv: str | Path,
    ffprobe_path: str | Path | None = None,
    ffmpeg_path: str | Path | None = None,
    segment_id: str = "main",
    ffprobe_timeout_s: float = 300.0,
    ffmpeg_timeout_s: float = 60.0,
) -> ManualPanelAnnotationTool:
    """Create and display the optional notebook annotation widget."""
    tool = ManualPanelAnnotationTool(
        video_pairs,
        annotation_csv=annotation_csv,
        ffprobe_path=ffprobe_path,
        ffmpeg_path=ffmpeg_path,
        segment_id=segment_id,
        ffprobe_timeout_s=ffprobe_timeout_s,
        ffmpeg_timeout_s=ffmpeg_timeout_s,
    )
    return tool.display()
