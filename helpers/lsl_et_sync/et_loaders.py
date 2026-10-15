"""Load gaze and IMU samples from native Tobii recordings.

Dependencies: gazedata.gz and imudata.gz, NumPy and pandas."""

import gzip
import json
import os
import zlib

import numpy as np
import pandas as pd

GZ_READ_ERRORS = (OSError, EOFError, zlib.error)

GAZE_FILE = 'gazedata.gz'
IMU_FILE = 'imudata.gz'


def safe_get(val, idx):
    if not isinstance(val, (list, str)):
        return np.nan
    return (json.loads(val) if isinstance(val, str) else val)[idx]


def _read_gz_lines(recording_path, file_name):
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
    lines = _read_gz_lines(recording_path, file_name)
    if lines is None:
        return None

    return pd.json_normalize([json.loads(line) for line in lines if line.strip()])


def load_et_recording(recording_path):
    raw_df = _read_gz_json(recording_path, GAZE_FILE)
    if raw_df is None:
        return None

    samples = []
    for values in raw_df.values:
        row = dict(zip(raw_df.columns, values))
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
    raw_df = _read_gz_json(recording_path, IMU_FILE)
    if raw_df is None:
        return None

    samples = []
    for values in raw_df.values:
        row = dict(zip(raw_df.columns, values))
        sample = {"timestamp": row["timestamp"]}
        for source, prefix in (("magnetometer", "magnetometer"),
                               ("accelerometer", "accelerometer"),
                               ("gyroscope", "gyro")):
            column = f"data.{source}"
            if column in row:
                sample.update({f"{prefix}.{axis}": safe_get(row[column], index)
                               for index, axis in enumerate(("x", "y", "z"))})
        samples.append(sample)

    return pd.DataFrame(samples)


