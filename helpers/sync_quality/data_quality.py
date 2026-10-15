"""Measure signal availability within driving modules in synchronized XDFs."""

import numpy as np
import pandas as pd

from helpers.sync_quality import report


METRICS = {
    'ET': ('gaze_available', 'pupil_available'),
    'Eyeball': ('direct_detection',),
    'DriverVideo': ('face_detected', 'pose_detected'),
}
REQUESTS = {
    'ET': ['Recording timestamp', 'Gaze point X', 'Gaze point Y',
           'Pupil diameter left', 'Pupil diameter right'],
    'ET_labels': ['Sensor', 'Validity left', 'Validity right'],
    'Eyeball': ['eyeball_timestamp', 'et_raw_timestamp',
                'Left_Pupil_Area', 'Right_Pupil_Area'],
    'Eyeball_labels': ['Left_Confidence', 'Right_Confidence'],
    'DriverVideo': ['driver_video_timestamp', 'et_raw_timestamp',
                    'face_detected', 'pose_detected'],
}


def count_availability(modality, frame, labels=None):
    if modality == 'ET':
        eye_rows = labels['Sensor'].eq('Eye Tracker').to_numpy()
        frame, labels = frame.loc[eye_rows], labels.loc[eye_rows]
        n_rows = len(frame)
        valid = np.column_stack([
            labels[f'Validity {side}'].eq('Valid').to_numpy()
            for side in ('left', 'right')])
        gaze = frame[['Gaze point X', 'Gaze point Y']].to_numpy(float)
        gaze_available = np.isfinite(gaze).all(axis=1) & valid.any(axis=1)
        pupils = frame[['Pupil diameter left', 'Pupil diameter right']].to_numpy(float)
        pupil_available = np.isfinite(pupils) & (pupils > 0) & valid
        return {'gaze_available': (int(gaze_available.sum()), n_rows),
                'pupil_available': (int(pupil_available.sum()), 2*n_rows)}
    if modality == 'Eyeball':
        areas = frame[['Left_Pupil_Area', 'Right_Pupil_Area']].to_numpy(float)
        confidence = np.column_stack([
            labels[f'{side}_Confidence'].fillna('').astype(str).str.strip().str.lower().to_numpy()
            for side in ('Left', 'Right')])
        positive = np.isfinite(areas) & (areas > 0)
        direct = positive & np.isin(confidence, ['high', 'medium'])
        total = 2*len(frame)
        return {'direct_detection': (int(direct.sum()), total)}
    if modality == 'DriverVideo':
        result = {}
        for name in ('face_detected', 'pose_detected'):
            values = pd.to_numeric(frame[name], errors='coerce').to_numpy(float)
            if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
                raise ValueError(f'{name} must contain finite 0/1 detection flags.')
            result[name] = (int((values == 1).sum()), len(frame))
        return result
    raise ValueError(f'Unknown modality: {modality}')


def build_data_quality(preprocessed_data_root, qc_tables, *, project, progress=True):
    recordings = qc_tables['et_lsl_alignment_csv'].sort_values(['pid', 'visit'])
    if recordings.duplicated(['pid', 'visit']).any():
        raise ValueError('Multiple QC rows refer to the same synchronized XDF.')
    rows = []
    for index, (_, recording) in enumerate(recordings.iterrows(), start=1):
        identity = report.record_id(recording, project)
        path = report.processed_xdf_path(preprocessed_data_root, recording)
        if progress:
            print(f'Data quality {index}/{len(recordings)}: {path.stem}', flush=True)
        streams = report.read_processed_xdf(preprocessed_data_root, recording, REQUESTS) if path.is_file() else {}
        scope = report.read_module_scope(preprocessed_data_root, recording) if path.is_file() else report.prepare_module_scope(None)
        for modality, mapping_key in (
                ('ET', 'et_lsl_alignment_csv'), ('Eyeball', 'et_eyeball_alignment_csv'),
                ('DriverVideo', 'driver_et_alignment_csv')):
            frame = streams.get(modality)
            label_name = modality + '_labels' if modality != 'DriverVideo' else None
            labels = streams.get(label_name) if label_name else None
            counts = {metric: (np.nan, np.nan) for metric in METRICS[modality]}
            status = 'assessed'
            if not path.is_file():
                status = 'xdf_absent'
            elif frame is None or frame.empty:
                status = 'stream_absent'
            elif label_name and labels is None:
                status = 'labels_absent'
            elif scope['status'] == 'unavailable':
                status = 'scope_unavailable'
            else:
                if label_name and not np.array_equal(
                        frame['__lsl_time_s'].to_numpy(), labels['__lsl_time_s'].to_numpy(),
                        equal_nan=True):
                    raise ValueError(f'{path.name}: {modality} numeric/label rows do not match.')
                mapping = report.matching_row(qc_tables[mapping_key], identity)
                if mapping is None:
                    mapping = pd.Series(dtype=object)
                times = report.scope_times(frame, modality, mapping, recording)
                mask, _ = report.module_membership(times, scope)
                if not mask.any():
                    status = 'no_module_rows'
                else:
                    counts = count_availability(modality, frame.loc[mask],
                                                labels.loc[mask] if labels is not None else None)
                    if not any(total for _, total in counts.values()):
                        status = 'no_sensor_rows'
            for metric, (count, total) in counts.items():
                rows.append(dict(modality=modality, metric=metric, status=status,
                                 percentage=100*count/total if total > 0 else np.nan))
    return pd.DataFrame(rows)
