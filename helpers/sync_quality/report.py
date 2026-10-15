"""Summarize synchronized timestamp coverage and file availability."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyxdf
from helpers.project_utils.helper_utils import ScenarioIDs

TASK_SCENARIO_IDS = tuple(ScenarioIDs)

_XDF_HEADER_CACHE = {}

def processed_xdf_path(preprocessed_data_root, record):
    return (Path(preprocessed_data_root) / 'XDF') / f"{record['pid']}_{record['visit']}_synchronized.xdf"


def read_processed_xdf(preprocessed_data_root, record, requested, *, synchronize_clocks=False,
                       dejitter_timestamps=False, allow_missing_channels=False):
    path = processed_xdf_path(preprocessed_data_root, record)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _XDF_HEADER_CACHE:
        _XDF_HEADER_CACHE[key] = pyxdf.resolve_streams(str(path))
    selected = [s for s in _XDF_HEADER_CACHE[key] if s['name'] in requested]
    if not selected:
        return {}
    columns_by_id = {}
    def take_columns(values, timestamps, header, stream_id):
        name = header['info']['name'][0]
        if stream_id not in columns_by_id:
            channels = header['info']['desc'][0]['channels'][0]['channel']
            labels = [c['label'][0] for c in channels]
            columns = [c for c in requested[name] if c in labels]
            columns_by_id[stream_id] = (columns, [labels.index(c) for c in columns])
        columns, indices = columns_by_id[stream_id]
        return np.asarray(values)[:, indices].copy(), timestamps, header
    streams, _ = pyxdf.load_xdf(
        str(path), select_streams=[s['stream_id'] for s in selected], on_chunk=take_columns,
        synchronize_clocks=synchronize_clocks, dejitter_timestamps=dejitter_timestamps, verbose=False)
    result = {}
    for stream in streams:
        name = stream['info']['name'][0]
        columns = columns_by_id.get(stream['info']['stream_id'], ([], []))[0]
        times = np.asarray(stream['time_stamps'], dtype=float)
        values = np.asarray(stream['time_series']) if len(times) else np.empty((0, len(columns)))
        if not allow_missing_channels and set(columns) != set(requested[name]):
            raise ValueError(f"{path.name}: missing requested channels in {name}")
        if name in result:
            raise ValueError(f"{path.name}: more than one {name} stream")
        result[name] = pd.DataFrame(values, columns=columns)
        result[name].insert(0, '__lsl_time_s', times)
    return result


PAIR_QC_TABLES = {
    'et_lsl': ('et_lsl_alignment_csv', 'ET'),
    'eyeball_et': ('et_eyeball_alignment_csv', 'Eyeball'),
    'idun_lsl': ('idun_lsl_alignment_csv', 'IDUN'),
    'driver_et': ('driver_et_alignment_csv', 'DriverVideo'),
}
IDENTITY = ['project', 'pid', 'visit', 'segment_id']

def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan

def flag(value):
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        return {'true': True, '1': True, 'false': False, '0': False}.get(value.strip().lower(), pd.NA)
    return bool(value)

def is_true(value):
    return flag(value) is True

def record_id(row, project='ImBrAid'):
    return dict(project=row.get('project') or project,
                pid=row.get('pid', row.get('participant')), visit=row['visit'],
                segment_id=row.get('segment_id') or 'main')

def matching_row(table, identity):
    if table.empty:
        return None
    match = table
    for field, value in identity.items():
        column = 'participant' if field == 'pid' and 'pid' not in table else field
        if column in match:
            match = match[match[column].replace('', 'main' if field == 'segment_id' else '').eq(value)]
        elif field == 'segment_id' and value != 'main':
            return None
    if len(match) > 1:
        raise ValueError(f'Multiple rows match {identity}')
    return match.iloc[0] if len(match) else None

def summarize_mappings(qc_tables, coverage, project='ImBrAid'):
    rows = []
    for pair, (qc_table_key, _) in PAIR_QC_TABLES.items():
        for _, row in qc_tables[qc_table_key].iterrows():
            rows.append(dict(
                **record_id(row, project), pair=pair,
                usable=flag(row['usable']),
                review_required=flag(row.get('review_required')),
            ))
    records = pd.DataFrame(rows, columns=IDENTITY + ['pair', 'usable', 'review_required'])
    for field in ('usable', 'review_required'):
        records[field] = records[field].astype('boolean')
    return records


def et_lsl_overlap(row):
    a, b = number(row['intercept_s']), number(row['scale'])
    start, end = number(row['et_start_s']), number(row['et_end_s'])
    lsl_start, lsl_end = number(row['lsl_start_s']), number(row['lsl_end_s'])
    if not np.isfinite([a, b, start, end, lsl_start, lsl_end]).all() or b <= 0:
        return np.nan, np.nan
    return max(start, (lsl_start - a) / b), min(end, (lsl_end - a) / b)

def prepare_module_scope(silab):
    scope = {
        'status': 'unavailable',
        'reason': '',
        'times': np.empty(0, dtype=float),
        'scenario': np.empty(0, dtype=float),
        'n_modules_found': 0,
    }
    if silab is None or not isinstance(silab, pd.DataFrame) or silab.empty:
        scope['reason'] = 'SILAB stream is missing or empty.'
        return scope
    time_column = next((column for column in ('__lsl_time_s', 'lsl_time_s')
                        if column in silab.columns), None)
    if time_column is None or 'ScenarioID' not in silab.columns:
        scope['reason'] = 'SILAB timestamps or ScenarioID channel are missing.'
        return scope
    times = pd.to_numeric(silab[time_column], errors='coerce').to_numpy(
        dtype=float, na_value=np.nan)
    scenario = pd.to_numeric(silab['ScenarioID'], errors='coerce').to_numpy(
        dtype=float, na_value=np.nan)
    if not np.isfinite(times).all():
        scope['reason'] = 'SILAB contains missing or nonfinite timestamps.'
        return scope
    if not np.isfinite(scenario).all():
        scope['reason'] = 'SILAB contains missing or nonfinite ScenarioID values.'
        return scope

    order = np.argsort(times, kind='stable')
    times, scenario = times[order], np.floor(scenario[order])
    task_ids = np.asarray(TASK_SCENARIO_IDS, dtype=float)
    present_ids = set(scenario).intersection(task_ids)
    missing_ids = [int(value) for value in task_ids if value not in present_ids]
    scope.update({
        'status': ('unavailable' if not present_ids else 'partial' if missing_ids else 'complete'),
        'reason': (f'{len(present_ids)}/{len(task_ids)} driving modules identified; '
                   f'missing ScenarioIDs: {", ".join(map(str, missing_ids))}.'
                   if missing_ids else
                   f'All {len(task_ids)} driving modules identified from SILAB ScenarioID.'),
        'times': times,
        'scenario': scenario,
        'n_modules_found': len(present_ids),
    })
    return scope


def module_membership(times, scope):
    times = np.asarray(times, dtype=float)
    finite = np.isfinite(times)
    member = np.zeros(times.shape, dtype=bool)
    unknown = ~finite
    if scope is None or scope.get('status') == 'unavailable':
        return member, unknown
    silab_times = np.asarray(scope.get('times', []), dtype=float)
    scenario = np.asarray(scope.get('scenario', []), dtype=float)
    if not len(silab_times) or len(scenario) != len(silab_times):
        return member, unknown
    covered = finite & (times >= silab_times[0]) & (times <= silab_times[-1])
    previous = np.searchsorted(silab_times, times[covered], side='right') - 1
    member[covered] = np.isin(scenario[previous], TASK_SCENARIO_IDS)
    return member, unknown

def read_module_scope(preprocessed_data_root, recording):
    streams = read_processed_xdf(
        preprocessed_data_root, recording, {'SILAB': ['ScenarioID']},
        synchronize_clocks=True, dejitter_timestamps=True, allow_missing_channels=True)
    return prepare_module_scope(streams.get('SILAB'))


def scope_times(frame, modality, mapping, et):
    times = frame['__lsl_time_s'].to_numpy(float).copy()
    if not len(times) or np.isfinite(times).all():
        return times
    ea, eb = number(et.get('intercept_s')), number(et.get('scale'))
    if modality == 'ET':
        predicted = ea + eb * frame['Recording timestamp'].to_numpy(float) / 1e6
    elif modality == 'Eyeball':
        et_time = frame['et_raw_timestamp'].to_numpy(float).copy()
        a, b = number(mapping.get('intercept_s')), number(mapping.get('scale'))
        if np.isfinite([a, b]).all() and b > 0:
            absent = ~np.isfinite(et_time)
            et_time[absent] = (frame['eyeball_timestamp'].to_numpy(float)[absent] - a) / b
        predicted = ea + eb * et_time
    elif modality == 'DriverVideo':
        a, b = number(mapping.get('driver_to_et_raw_intercept_s')), number(mapping.get('driver_to_et_raw_scale'))
        predicted = ea + eb * (a + b * frame['driver_video_timestamp'].to_numpy(float))
    else:
        a, b = number(mapping.get('intercept_s')), number(mapping.get('scale'))
        native = frame['idun_timestamp'].to_numpy(float) - number(mapping.get('idun_raw_time_origin_unix_s'))
        predicted = a + b * native
    absent = ~np.isfinite(times)
    times[absent] = predicted[absent]
    return times


def build_coverage(preprocessed_data_root, qc_tables, scope='recording', *, project):
    if scope not in ('recording', 'modules'):
        raise ValueError("scope must be 'recording' or 'modules'")
    requests = {'ET': ['Recording timestamp'], 'Eyeball': ['et_raw_timestamp'],
                'DriverVideo': ['driver_video_timestamp', 'et_raw_timestamp'],
                'IDUN': ['idun_timestamp', 'et_raw_timestamp']}
    if scope == 'modules':
        requests['Eyeball'].append('eyeball_timestamp')
    rows = []
    for _, recording in qc_tables['et_lsl_alignment_csv'].sort_values(['pid', 'visit']).iterrows():
        identity, path = record_id(recording, project), processed_xdf_path(preprocessed_data_root, recording)
        streams = read_processed_xdf(preprocessed_data_root, recording, requests)
        module_scope = read_module_scope(preprocessed_data_root, recording) if scope == 'modules' else None
        et = recording
        et_ok = is_true(et['usable'])
        et_low, et_high = et_lsl_overlap(et) if et_ok else (np.nan, np.nan)
        for modality, mapping_key in [('ET', 'et_lsl_alignment_csv'), ('Eyeball', 'et_eyeball_alignment_csv'),
                                     ('DriverVideo', 'driver_et_alignment_csv'), ('IDUN', 'idun_lsl_alignment_csv')]:
            mapping = matching_row(qc_tables[mapping_key], identity)
            frame = streams.get(modality, pd.DataFrame(columns=['__lsl_time_s', *requests[modality]]))
            if len(frame) and (mapping is None or not is_true(mapping['usable'])):
                raise ValueError(f"{path.name}: {modality} stream has no usable QC mapping")
            stream_rows = len(frame)
            scope_unknown = 0
            if scope == 'modules':
                mask, unknown = module_membership(scope_times(frame, modality, mapping, et), module_scope)
                scope_unknown = int(unknown.sum())
                frame = frame.loc[mask]
            for clock in (('lsl',) if modality == 'ET' else ('et_raw', 'lsl')):
                low = high = np.nan
                if modality == 'ET':
                    native = frame['Recording timestamp'].to_numpy(float) / 1e6
                    low, high = et_low, et_high
                elif modality == 'Eyeball':
                    native = frame['et_raw_timestamp'].to_numpy(float)
                    if mapping is not None and is_true(mapping['usable']):
                        low, high = number(mapping['et_start_s']), number(mapping['et_end_s'])
                        if clock == 'lsl':
                            low, high = (max(low, et_low), min(high, et_high)) if et_ok else (np.nan, np.nan)
                elif modality == 'DriverVideo':
                    native = frame['driver_video_timestamp'].to_numpy(float)
                    if mapping is not None and is_true(mapping['usable']):
                        low, high = number(mapping['driver_interpolation_start_s']), number(mapping['driver_interpolation_end_s'])
                        if clock == 'lsl':
                            driver_to_et_intercept = number(mapping['driver_to_et_raw_intercept_s'])
                            driver_to_et_scale = number(mapping['driver_to_et_raw_scale'])
                            if et_ok:
                                low = max(0, low, (et_low - driver_to_et_intercept) / driver_to_et_scale)
                                high = min(high, (et_high - driver_to_et_intercept) / driver_to_et_scale)
                            else:
                                low, high = np.nan, np.nan
                else:
                    origin = number(mapping.get('idun_raw_time_origin_unix_s')) if mapping is not None else np.nan
                    native = frame['idun_timestamp'].to_numpy(float) - origin
                    if mapping is not None and is_true(mapping['usable']):
                        low, high = number(mapping['fit_support_start_idun_raw_s']), number(mapping['fit_support_end_idun_raw_s'])
                        if clock == 'et_raw':
                            idun_to_lsl_intercept = number(mapping['intercept_s'])
                            idun_to_lsl_scale = number(mapping['scale'])
                            et_to_lsl_intercept = number(et['intercept_s'])
                            et_to_lsl_scale = number(et['scale'])
                            if et_ok:
                                low = max(low, (et_to_lsl_intercept + et_to_lsl_scale * et_low
                                                - idun_to_lsl_intercept) / idun_to_lsl_scale)
                                high = min(high, (et_to_lsl_intercept + et_to_lsl_scale * et_high
                                                  - idun_to_lsl_intercept) / idun_to_lsl_scale)
                            else:
                                low, high = np.nan, np.nan
                target = frame['__lsl_time_s' if clock == 'lsl' else 'et_raw_timestamp'].to_numpy(float)
                finite = np.isfinite(target)
                supported = finite & np.isfinite(native) & (native >= low) & (native <= high)
                if modality == 'IDUN' and clock == 'et_raw' and len(frame):
                    if et_ok:
                        within_idun_support = (
                            (native >= number(mapping['fit_support_start_idun_raw_s']))
                            & (native <= number(mapping['fit_support_end_idun_raw_s'])))
                        within_et_support = (
                            (target >= number(et['et_start_s']))
                            & (target <= number(et['et_end_s'])))
                        within_lsl_support = (
                            (frame['__lsl_time_s'].to_numpy(float) >= number(et['lsl_start_s']))
                            & (frame['__lsl_time_s'].to_numpy(float) <= number(et['lsl_end_s'])))
                        supported = finite & within_idun_support & within_et_support & within_lsl_support
                    else:
                        supported = np.zeros(len(frame), dtype=bool)
                total, n_finite, n_supported = len(frame), int(finite.sum()), int(supported.sum())
                if module_scope is not None and module_scope['status'] == 'unavailable':
                    total = n_finite = n_supported = np.nan
                rows.append(dict(**identity, modality=modality, clock=clock,
                                 n_stream_rows=stream_rows, coverage_scope=scope,
                                 module_scope_status=module_scope['status'] if module_scope else 'not_requested',
                                 scope_reason=module_scope['reason'] if module_scope else '',
                                 n_modules_found=module_scope['n_modules_found'] if module_scope else np.nan,
                                 n_scope_unknown=scope_unknown,
                                 n_total=total, n_finite=n_finite, n_supported=n_supported,
                                 n_extrapolated=n_finite-n_supported, n_missing=total-n_finite,
                                 support_start_s=low, support_end_s=high))
    return pd.DataFrame(rows)


def load_qc_tables(qc_table_paths):
    paths = {key: Path(path) for key, path in qc_table_paths.items()
             if key.endswith('_alignment_csv')}
    tables = {key: pd.read_csv(path, dtype={k: str for k in
              ['project', 'pid', 'participant', 'visit', 'segment_id']}, keep_default_na=False)
              for key, path in paths.items()}
    return paths, tables
def build_data_availability(preprocessed_data_root, recordings):
    root = Path(preprocessed_data_root)
    time_columns = ['et_raw_timestamp', 'lsl_timestamp']
    modalities = [
        ('Eyeball', 'eyeball', '_eyeball_synchronized.csv'),
        ('DriverVideo', 'driver_video', '_driver_video_synchronized.csv'),
        ('IDUN', 'idun', '_idun_synchronized.csv'),
    ]
    rows = []
    identities = recordings[['pid', 'visit']].drop_duplicates().sort_values(['pid', 'visit'])
    for pid, visit in identities.itertuples(index=False, name=None):
        row = {'pid': pid, 'visit': visit}
        name = f'{pid}_{visit}'
        for modality, prefix, suffix in modalities:
            path = root / modality / f'{name}{suffix}'
            row[f'{prefix}_available'] = path.is_file()
            finite = np.zeros(2, dtype=bool)
            if path.is_file():
                with pd.read_csv(path, usecols=time_columns, chunksize=1000) as chunks:
                    for chunk in chunks:
                        finite |= np.isfinite(chunk[time_columns].to_numpy(dtype=float)).any(axis=0)
                        if finite.all():
                            break
            row[f'{prefix}_et_available'] = bool(finite[0])
            row[f'{prefix}_lsl_available'] = bool(finite[1])
        row['et_exported_available'] = (root / 'ET' / f'{name}.tsv').is_file()
        rows.append(row)
    return pd.DataFrame(rows)


def build_user_summary(preprocessed_data_root, qc_tables, coverage, *, project):
    columns = ['Recording ID', 'Data type', 'Data available', 'Time alignment', 'Use note']
    mapping_keys = {
        'ET': 'et_lsl_alignment_csv',
        'Eyeball': 'et_eyeball_alignment_csv',
        'DriverVideo': 'driver_et_alignment_csv',
        'IDUN': 'idun_lsl_alignment_csv',
    }
    modalities = {
        'ET': ('Eye tracking', '.tsv', 'TSV'),
        'Eyeball': ('Eyeball pupil', '_eyeball_synchronized.csv', 'CSV'),
        'DriverVideo': ('Driver landmarks', '_driver_video_synchronized.csv', 'CSV'),
        'IDUN': ('Ear EEG', '_idun_synchronized.csv', 'CSV'),
    }
    required_counts = ['n_total', 'n_finite', 'n_supported', 'n_extrapolated', 'n_missing']
    required_coverage = ['pid', 'visit', 'modality', 'clock', *required_counts]
    missing_columns = set(required_coverage).difference(coverage.columns)
    if missing_columns:
        raise ValueError(f'Coverage is missing columns: {sorted(missing_columns)}')

    for key in mapping_keys.values():
        if key not in qc_tables:
            raise ValueError(f'Missing QC table: {key}')
        identities = [record_id(row, project) for _, row in qc_tables[key].iterrows()]
        seen = set()
        for identity in identities:
            identity_key = tuple(identity.get(field) for field in IDENTITY)
            if not identity.get('pid') or not identity.get('visit'):
                raise ValueError(f'{key}: missing recording identity')
            if identity_key in seen:
                raise ValueError(f'{key}: duplicate recording identity {identity}')
            seen.add(identity_key)

    coverage_identities = set()
    for _, row in coverage.iterrows():
        identity = record_id(row, project)
        identity_key = (*[identity.get(field) for field in IDENTITY], row['modality'], row['clock'])
        if identity_key in coverage_identities:
            raise ValueError(f'Duplicate coverage identity: {identity}, {row["modality"]}, {row["clock"]}')
        coverage_identities.add(identity_key)
        counts = {field: number(row[field]) for field in required_counts}
        if row.get('coverage_scope') == 'modules' and row.get('module_scope_status') == 'unavailable':
            if any(np.isfinite(value) for value in counts.values()):
                raise ValueError(f'Unavailable module coverage must have missing counts: {identity}')
            continue
        if any(not np.isfinite(value) or value < 0 or value != int(value) for value in counts.values()):
            raise ValueError(f'Coverage counts must be nonnegative integers: {identity}')
        if (counts['n_finite'] + counts['n_missing'] != counts['n_total']
                or counts['n_supported'] + counts['n_extrapolated'] != counts['n_finite']):
            raise ValueError(f'Coverage counts are inconsistent: {identity}')

    def percentage(part, total):
        value = 100 * part / total
        return '<0.1%' if 0 < value < 0.05 else f'{value:.1f}%'

    def requires_review(mapping):
        return (is_true(mapping.get('review_required'))
                or 'review' in str(mapping.get('mapping_status', '')).lower())

    publication_root = Path(preprocessed_data_root)
    rows = []
    for _, recording in qc_tables['et_lsl_alignment_csv'].sort_values(['pid', 'visit']).iterrows():
        identity = record_id(recording, project)
        recording_name = f'{identity["pid"]}_{identity["visit"]}'
        display_id = recording_name
        if identity.get('segment_id', 'main') != 'main':
            display_id += f' ({identity["segment_id"]})'
        for modality, (label, suffix, file_kind) in modalities.items():
            lsl_rows = coverage.loc[coverage['modality'].eq(modality) & coverage['clock'].eq('lsl')]
            cov = matching_row(lsl_rows, identity)
            file_available = (publication_root / modality / f'{recording_name}{suffix}').is_file()
            stream_available = cov is not None and number(cov.get('n_stream_rows', cov['n_total'])) > 0
            if stream_available and file_available:
                availability = f'XDF + {file_kind}'
            elif stream_available:
                availability = 'XDF only'
            elif file_available:
                availability = f'{file_kind} only'
            else:
                availability = 'Not provided'

            notes = []
            if not stream_available:
                alignment = 'Not assessed' if file_available or cov is None else 'Not applicable'
                if file_available and cov is not None:
                    notes.append(f'{file_kind} provided; no corresponding stream in synchronized XDF.')
                if cov is None:
                    notes.append('XDF timestamp coverage unavailable.')
            else:
                mapping = matching_row(qc_tables[mapping_keys[modality]], identity)
                components = [mapping]
                if modality in ('Eyeball', 'DriverVideo'):
                    components.append(matching_row(qc_tables['et_lsl_alignment_csv'], identity))
                if any(component is None or not is_true(component.get('usable')) for component in components):
                    raise ValueError(f'{display_id}: {modality} XDF stream has no usable LSL mapping chain')
                review = any(requires_review(component) for component in components)
                module_mode = cov.get('coverage_scope') == 'modules'
                scope_unavailable = module_mode and cov.get('module_scope_status') == 'unavailable'
                row_label = 'driving-module samples' if module_mode else 'XDF rows'
                if scope_unavailable:
                    alignment = 'LSL; coverage not assessed'
                    notes.append('Driving-module coverage unavailable. ' + str(cov.get('scope_reason', '')))
                else:
                    total = int(number(cov['n_total']))
                    finite = int(number(cov['n_finite']))
                    missing = int(number(cov['n_missing']))
                    outside = int(number(cov['n_extrapolated']))
                    if total == 0:
                        alignment = 'LSL; no module samples' if module_mode else 'No LSL timestamps'
                        notes.append('No samples in the identified driving modules.' if module_mode else 'No XDF timestamp rows.')
                    elif finite == 0:
                        alignment = 'No LSL timestamps'
                        notes.append('All driving-module XDF timestamps are missing.' if module_mode else 'All XDF timestamps are missing.')
                    else:
                        alignment = 'Partial LSL' if missing else 'LSL'
                        single_anchor = modality == 'DriverVideo' and (
                            number(mapping.get('anchor_count')) == 1
                            or str(mapping.get('mapping_status', '')).lower() == 'offset_only')
                        if single_anchor:
                            notes.append('One timing event. Clock drift was not estimated.')
                        else:
                            lower = number(cov.get('support_start_s'))
                            upper = number(cov.get('support_end_s'))
                            if not np.isfinite([lower, upper]).all() or upper < lower:
                                notes.append('Supported interval unavailable.')
                            elif outside:
                                notes.append(f'{percentage(outside, total)} of {row_label} outside timing support.')
                        if missing:
                            notes.append(f'{percentage(missing, total)} of {row_label} have missing timestamps.')
                    if module_mode and number(cov.get('n_scope_unknown', 0)) > 0:
                        notes.append(f"{int(number(cov['n_scope_unknown']))} samples could not be assigned to a module; excluded from percentages.")
                if review:
                    alignment += '; review required'
                if any(pd.isna(flag(component.get('review_required'))) and not requires_review(component)
                       for component in components):
                    notes.append('Timing review status unavailable.')
            if (cov is not None and (stream_available or file_available)
                    and cov.get('coverage_scope') == 'modules'
                    and cov.get('module_scope_status') == 'partial'):
                notes.append(f"Only {int(number(cov['n_modules_found']))}/10 driving modules identified.")
            rows.append(dict(zip(columns, [display_id, label, availability, alignment, ' '.join(notes)])))
    return pd.DataFrame(rows, columns=columns)

def load_or_build_coverage(preprocessed_data_root, qc_tables, scope='modules', *,
                           project, cache_path, force=False, progress=True):
    cache_path = Path(cache_path)
    reason = 'forced refresh' if force else 'cache missing'
    if not force:
        try:
            data = json.loads(cache_path.read_text(encoding='utf-8'))
            coverage = pd.DataFrame(data['data'], columns=data['columns'])
            return coverage, {
                'status': 'cache hit', 'reason': 'saved coverage', 'path': str(cache_path),
            }
        except FileNotFoundError:
            reason = 'cache missing'
        except (OSError, UnicodeError, ValueError, TypeError, KeyError):
            reason = 'cache unreadable'
    if progress:
        print(f'Coverage: {reason}; reading XDFs once to build the shared cache.')
    coverage = build_coverage(preprocessed_data_root, qc_tables, scope=scope, project=project)
    data = json.loads(coverage.to_json(orient='split', index=False, double_precision=15))
    temporary = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=cache_path.parent,
            prefix=cache_path.name + '.', suffix='.tmp', delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, cache_path)
        temporary = None
    except OSError as error:
        return coverage, {
            'status': 'computed; cache not saved', 'reason': str(error), 'path': str(cache_path),
        }
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return coverage, {
        'status': 'computed and cached', 'reason': reason, 'path': str(cache_path),
    }
