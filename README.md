# ImBrAid processing code

Notebooks and shared helpers for anonymization, device preprocessing, synchronization, release preparation and technical validation.

## Setup and configuration

Use one Python environment for the notebook kernel and install the packages in [ImBrAid/requirements.txt](ImBrAid/requirements.txt):

```shell
python -m pip install -r ImBrAid/requirements.txt
```

Open notebooks in Jupyter or another notebook frontend, with the working directory inside this code directory. FFmpeg and ffprobe must also be installed and available on PATH. Driver landmark extraction downloads missing MediaPipe task models from the URLs in its settings cell.

[ImBrAid/config.json](ImBrAid/config.json) is the shared configuration. Relative paths are resolved from its containing directory. The supplied configuration uses test locations under `E:/test`; update these before using another dataset.

| Path key | Role |
|---|---|
| `raw_data_root` | Original identifiable input for anonymization; requires `T###_V#` identities. Its current published-raw setting must be changed for a fresh anonymization run. |
| `random_id_root` | Copy with randomized recording IDs. |
| `processing_data_root` | Time-shifted copy used by preprocessing and synchronization. |
| `eyeball_work_root` | Converted eyeball MP4s and pupil-feature CSVs. |
| `driver_landmark_root` | Driver landmark CSVs and downloaded models. |
| `idun_repair_root` | Reconstructed IDUN CSVs, chunk mappings and repair QC. |
| `publication_root` | Released CSVs, manually staged ET TSVs and synchronized XDFs. |
| `publication_raw_data_root` | Raw release copy produced by the packaging notebook. |
| `not_public_controled_data_root` | Controlled-access videos and raw ET recording directories. |

The `files` section sets the manual-annotation file, alignment/QC/export-summary CSVs and `private_mapping_csv` (the private original-to-random-ID lookup). The `modalities` section sets source subfolder names and the cap-EEG stream name.

## Order and notebook inputs/outputs

Paths below refer to config keys. `<pid>` is a randomized participant-visit record ID; the anonymized visit label is normally `VX`.

1. For preparation from original recordings, run anonymization and time shifting.
2. Run the three device `00` notebooks in any order, then Driver `00.5`.
3. Run SyncQuality `01`–`05`; review manual annotations in `02` and IDUN repair QC in `04`.
4. Run `10`. Before `11`, prepare the raw release with `10.5` and manually place Tobii Pro Lab exports at `publication_root/ET/<pid>_<visit>.tsv`.
5. Run `11`, then `12` and `13`. `14` can run whenever the published IDUN quality CSVs are available.

Existing time-shifted inputs allow starting at device preprocessing. Existing release files allow running only the relevant validation notebooks.

| Notebook | Purpose / input | Output |
|---|---|---|
| [Anonymization/run_only_once.ipynb](ImBrAid/Anonymization/run_only_once.ipynb) | Assign persistent random IDs to original `raw_data_root` records; optionally shift calendar times. | `random_id_root`, `private_mapping_csv` and `processing_data_root`. |
| [Eyeball/00_eyeball_pipeline.ipynb](ImBrAid/Eyeball/00_eyeball_pipeline.ipynb) | Convert eyeball WEBMs and extract bilateral pupil features from `processing_data_root`. | `eyeball_work_root/mp4_videos/*.mp4` and `pupil_sizes/*_pupil_sizes.csv`. |
| [DriverVideo/00_extract_driver_landmarks.ipynb](ImBrAid/DriverVideo/00_extract_driver_landmarks.ipynb) | Extract face/pose landmarks from source Driver Video recordings. | `driver_landmark_root/*_face_pose_landmarks.csv`; models under `models/`. |
| [DriverVideo/00.5_persist_exact_driver_pts.ipynb](ImBrAid/DriverVideo/00.5_persist_exact_driver_pts.ipynb) | Read exact source-video frame PTS with ffprobe after Driver `00`. | Adds `driver_video_time_s` to the existing landmark CSVs. |
| [IDUN/00_idun_raw_lsl_repair.ipynb](ImBrAid/IDUN/00_idun_raw_lsl_repair.ipynb) | Reconstruct native IDUN sample order from EEG CSVs and XDF chunks in `processing_data_root`. | `idun_repair_root/*_idun_repaired.csv`, `chunks/*_idun_chunk_mapping.csv`, `idun_repair_summary.csv` and `idun_repair_issues.csv`. |
| [SyncQuality/01_et_raw_lsl_alignment.ipynb](ImBrAid/SyncQuality/01_et_raw_lsl_alignment.ipynb) | Fit raw ET-clock → LSL mappings using gaze/IMU signals in raw ET recordings and XDFs. | Configured `et_lsl_*` alignment, window and issue CSVs. |
| [SyncQuality/02_driver_et_raw_alignment.ipynb](ImBrAid/SyncQuality/02_driver_et_raw_alignment.ipynb) | Manually annotate shared panel transitions in Driver Video and ET scene video; fit Driver PTS → ET mappings. | Updated annotation CSV; `driver_et_alignment_csv` and `driver_et_issues_csv`. |
| [SyncQuality/03_eyeball_et_raw_alignment.ipynb](ImBrAid/SyncQuality/03_eyeball_et_raw_alignment.ipynb) | Align extracted Eyeball pupil features with raw Tobii pupil traces. | Configured `et_eyeball_*` alignment, window and issue CSVs. |
| [SyncQuality/04_idun_raw_repair_qc.ipynb](ImBrAid/SyncQuality/04_idun_raw_repair_qc.ipynb) | Review the IDUN repair summary and issues from IDUN `00`. | Notebook QC tables; no new data files. |
| [SyncQuality/05_idun_raw_lsl_alignment.ipynb](ImBrAid/SyncQuality/05_idun_raw_lsl_alignment.ipynb) | Use repair-manifest source paths to align native IDUN EEG with cap EEG in XDF. | Configured `idun_lsl_*` alignment, channel, window and issue CSVs. |
| [SyncQuality/10_publication_export.ipynb](ImBrAid/SyncQuality/10_publication_export.ipynb) | Apply `01/02/03/05` mappings to pupil features, exact-PTS landmarks and native IDUN EEG. | `publication_root/{Eyeball,DriverVideo,IDUN}/*_synchronized.csv` and configured export-summary/issue CSVs. |
| [Anonymization/10.5_move_controlled_data.ipynb](ImBrAid/Anonymization/10.5_move_controlled_data.ipynb) | Copy `processing_data_root`; separate driver/eyeball videos and raw ET directories; remove IDUN PDFs from the release copy. | Rebuilt `publication_raw_data_root` and `not_public_controled_data_root`. |
| [SyncQuality/11_generate_xdf.ipynb](ImBrAid/SyncQuality/11_generate_xdf.ipynb) | Combine released raw XDFs, `10` CSVs and manually staged ET TSVs using the ET → LSL mapping. Before running, the exported ET tsv files should be copied to `publication_root/preprocessed_data/ET/`| `publication_root/XDF/<pid>_<visit>_synchronized.xdf`. |
| [SyncQuality/12_sync_quality_report.ipynb](ImBrAid/SyncQuality/12_sync_quality_report.ipynb) | Summarize saved QC and synchronized-XDF coverage; compare pupil signals and native IDUN EEG with cap EEG. | `et_lsl_alignment_csv.parent/12_sync_quality_report/`: report CSVs, PNG/SVG figures, example JSONs, source manifest and README. |
| [SyncQuality/13_technique_validation_01.ipynb](ImBrAid/SyncQuality/13_technique_validation_01.ipynb) | Compute module-level SDLP and eye-event duration measures from synchronized XDFs. | Notebook plots; no standalone files. |
| [SyncQuality/14_technique_validation_02.ipynb](ImBrAid/SyncQuality/14_technique_validation_02.ipynb) | Pool `publication_raw_data_root/<pid>/IDUN/*_quality.csv` signal-quality scores. | Notebook statistics and 5-point/1-point histograms. |

[SyncQuality/annotations/panel_annotations.csv](ImBrAid/SyncQuality/annotations/panel_annotations.csv) contains saved Driver/ET panel-on/off decisions and exact PTS brackets. Review or recreate these when recording IDs or source videos change. Saving a decision in notebook `02` updates this file.

### Reruns and path constraints

- `run_only_once` currently has `EXPORT=False` and `RUN_TIME_SHIFT=False`; enable the required action explicitly.
- Eyeball and Driver `00` skip existing outputs by default. IDUN `00`, Driver `00.5`, mapping/export notebooks and generated reports replace their named outputs on rerun.
- **`10.5` currently has `RUN=True` and deletes both existing raw/controlled destination trees before rebuilding them.** It preserves the processing source and does not stage ET TSVs.
- `11` reads raw XDFs from `publication_root.parent / 'raw_data'`. Keep this consistent with `publication_raw_data_root`. It replaces original IDUN streams; records without a synchronized IDUN CSV receive no replacement IDUN stream.
- Repair manifests and QC tables store source file paths. Regenerate upstream manifests/QC after relocating inputs. Keep source and output directories separate.

## Shared helper files

Helpers are imported by the notebooks; run the notebooks as the workflow entry points.

| File under `helpers/` | Role |
|---|---|
| [project_config.py](helpers/project_config.py) | Load shared JSON settings and resolve paths. |
| [helper_lsl.py](helpers/helper_lsl.py) | Select named XDF streams, choosing the longest when duplicated. |
| [project_utils/helper_utils.py](helpers/project_utils/helper_utils.py) | Shared ET feature groups, processing rate and driving-module IDs. |
| [anonymization/export.py](helpers/anonymization/export.py) | Discover records, manage random IDs and rewrite identifiable text/metadata during copying. |
| [anonymization/time_shift.py](helpers/anonymization/time_shift.py) | Shift calendar timestamps and remove simulator wall-clock fields in the processing copy. |
| [DriverVideo/driver_landmarks.py](helpers/DriverVideo/driver_landmarks.py) | Discover videos, obtain models and extract per-frame face/pose landmarks. |
| [Eyeball/webm_to_mp4.py](helpers/Eyeball/webm_to_mp4.py) | Convert eyeball WEBMs with FFmpeg. |
| [Eyeball/eye_detection.py](helpers/Eyeball/eye_detection.py) | Detect/track pupil ellipses and write pupil areas and detection confidence. |
| [Eyeball/blink.py](helpers/Eyeball/blink.py) | Detect bilateral blinks and summarize counts/durations. |
| [idun/raw_lsl_repair.py](helpers/idun/raw_lsl_repair.py) | Match EEG samples to XDF chunks and produce reconstruction diagnostics. |
| [lsl_et_sync/et_loaders.py](helpers/lsl_et_sync/et_loaders.py) | Read raw Tobii gzip gaze and IMU records. |
| [lsl_et_sync/lsl_loaders.py](helpers/lsl_et_sync/lsl_loaders.py) | Load native ET XDF streams and measure their recording coverage. |
| [lsl_et_sync/offset.py](helpers/lsl_et_sync/offset.py) | Estimate signal offsets and robust windowed affine clock mappings. |
| [sync_quality/et_lsl_alignment.py](helpers/sync_quality/et_lsl_alignment.py) | Discover inputs, fit ET → LSL mappings and produce QC. |
| [sync_quality/panel_annotation.py](helpers/sync_quality/panel_annotation.py) | Exact-PTS frame inspection and manual panel annotation widgets. |
| [sync_quality/driver_et_alignment.py](helpers/sync_quality/driver_et_alignment.py) | Fit and assess Driver → ET mappings from annotated transitions. |
| [sync_quality/driver_lsl_alignment.py](helpers/sync_quality/driver_lsl_alignment.py) | Compose Driver → ET and ET → LSL mappings. |
| [sync_quality/et_eyeball_alignment.py](helpers/sync_quality/et_eyeball_alignment.py) | Fit ET/Eyeball pupil-based mappings and local-window QC. |
| [sync_quality/eyeball_time_table.py](helpers/sync_quality/eyeball_time_table.py) | Export pupil tables with native, ET and LSL timestamps. |
| [sync_quality/driver_timestamp_export.py](helpers/sync_quality/driver_timestamp_export.py) | Persist exact frame PTS and export synchronized landmark CSVs. |
| [sync_quality/idun_lsl_alignment.py](helpers/sync_quality/idun_lsl_alignment.py) | Align native IDUN EEG to cap EEG and export synchronized samples. |
| [sync_quality/publication_contract.py](helpers/sync_quality/publication_contract.py) | Define and validate publication CSV schemas. |

The eight `__init__.py` files mark the helper packages. `anonymization/__init__.py` exposes its workflow functions; `DriverVideo`, `idun` and `sync_quality` declare their available modules.
