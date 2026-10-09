import gzip
import json
import os
import zlib

import numpy as np
import pandas as pd

# A recording that was cut off mid-write leaves a gz file that cannot be
# decompressed. That is a property of the recording, not a bug, so the loaders
# report it and return nothing instead of raising.
GZ_READ_ERRORS = (OSError, EOFError, zlib.error)

GAZE_FILE = 'gazedata.gz'
IMU_FILE = 'imudata.gz'


def safe_get(val, idx):
    """
    Return element `idx` of a list-valued cell, or NaN when there is none.

    The gz files store vectors either as a JSON string or as a list, and
    missing samples as a scalar NaN, hence the type check.
    """
    if not isinstance(val, (list, str)):
        return np.nan
    return (json.loads(val) if isinstance(val, str) else val)[idx]


def _read_gz_lines(recording_path, file_name):
    """
    Read the lines of one gz file of an ET recording.

    Returns None when the file is missing or cannot be decompressed, which
    happens when the recording was interrupted while it was being written.
    """
    path = os.path.join(recording_path, file_name)
    if not os.path.exists(path):
        return None

    try:
        with gzip.open(path, 'rb') as handle:
            return handle.read().decode('utf-8').splitlines()
    except GZ_READ_ERRORS as error:
        print(f"Cannot read {path}: {error}")
        return None


def _read_gz_json(recording_path, file_name):
    """
    Read one line-delimited JSON gz file of an ET recording into a DataFrame.

    Returns None when the file cannot be read; the callers then fall back to
    the other stream or skip the visit.
    """
    lines = _read_gz_lines(recording_path, file_name)
    if lines is None:
        return None

    return pd.json_normalize([json.loads(line) for line in lines if line.strip()])


def load_et_recording(recording_path):
    """Load gaze samples from one explicitly selected ET recording folder."""

    raw_df = _read_gz_json(recording_path, GAZE_FILE)
    if raw_df is None:
        return None

    samples = []
    for _, row in raw_df.iterrows():
        samples.append({
            "timestamp": row["timestamp"],
            "gaze2d.x": safe_get(row["data.gaze2d"], 0),
            "gaze2d.y": safe_get(row["data.gaze2d"], 1),
            "gaze3d.x": safe_get(row["data.gaze3d"], 0),
            "gaze3d.y": safe_get(row["data.gaze3d"], 1),
            "gaze3d.z": safe_get(row["data.gaze3d"], 2),
            "left-gaze-direction.x": safe_get(row["data.eyeleft.gazedirection"], 0),
            "left-gaze-direction.y": safe_get(row["data.eyeleft.gazedirection"], 1),
            "left-gaze-direction.z": safe_get(row["data.eyeleft.gazedirection"], 2),
            "left-gaze-origin.x": safe_get(row["data.eyeleft.gazeorigin"], 0),
            "left-gaze-origin.y": safe_get(row["data.eyeleft.gazeorigin"], 1),
            "left-gaze-origin.z": safe_get(row["data.eyeleft.gazeorigin"], 2),
            "left-pupil": row["data.eyeleft.pupildiameter"],
            "right-gaze-direction.x": safe_get(row["data.eyeright.gazedirection"], 0),
            "right-gaze-direction.y": safe_get(row["data.eyeright.gazedirection"], 1),
            "right-gaze-direction.z": safe_get(row["data.eyeright.gazedirection"], 2),
            "right-gaze-origin.x": safe_get(row["data.eyeright.gazeorigin"], 0),
            "right-gaze-origin.y": safe_get(row["data.eyeright.gazeorigin"], 1),
            "right-gaze-origin.z": safe_get(row["data.eyeright.gazeorigin"], 2),
            "right-pupil": row["data.eyeright.pupildiameter"],
        })

    return pd.DataFrame(samples)


def load_imu_recording(recording_path):
    """Load IMU samples from one explicitly selected ET recording folder."""

    raw_df = _read_gz_json(recording_path, IMU_FILE)
    if raw_df is None:
        return None

    samples = []
    for _, row in raw_df.iterrows():
        sample = {"timestamp": row["timestamp"]}
        if "data.magnetometer" in row:
            sample.update({
                "magnetometer.x": safe_get(row["data.magnetometer"], 0),
                "magnetometer.y": safe_get(row["data.magnetometer"], 1),
                "magnetometer.z": safe_get(row["data.magnetometer"], 2),
            })
        if "data.accelerometer" in row:
            sample.update({
                "accelerometer.x": safe_get(row["data.accelerometer"], 0),
                "accelerometer.y": safe_get(row["data.accelerometer"], 1),
                "accelerometer.z": safe_get(row["data.accelerometer"], 2),
            })
        if "data.gyroscope" in row:
            sample.update({
                "gyro.x": safe_get(row["data.gyroscope"], 0),
                "gyro.y": safe_get(row["data.gyroscope"], 1),
                "gyro.z": safe_get(row["data.gyroscope"], 2),
            })
        samples.append(sample)

    return pd.DataFrame(samples)


