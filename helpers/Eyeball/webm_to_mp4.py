import os

import ffmpeg

# Encoding settings: crf 15 is visually lossless, which matters because the
# pupil detection thresholds the raw pixel values.
VIDEO_CODEC = 'libx264'
CRF = 15
PRESET = 'fast'


def convert_webm_file(webm_path, mp4_path, overwrite=False):
    """
    Re-encode a single webm recording to H.264 mp4.

    Args:
        webm_path: the source .webm file.
        mp4_path: the .mp4 file to write; its folder is created if needed.
        overwrite: re-encode even when `mp4_path` already exists.

    Returns:
        True when the file was encoded, False when it already existed (and
        `overwrite` is False) or when ffmpeg reported an error. ffmpeg errors
        are printed rather than raised so a batch run keeps going.
    """
    if os.path.exists(mp4_path) and not overwrite:
        return False

    os.makedirs(os.path.dirname(os.path.abspath(mp4_path)), exist_ok=True)
    try:
        (
            ffmpeg
            .input(webm_path)
            .output(
                mp4_path,
                vcodec=VIDEO_CODEC,
                crf=CRF,
                preset=PRESET,
            )
            .run(overwrite_output=True)
        )
    except ffmpeg.Error as error:
        print('Error', error.stderr.decode('utf8') if error.stderr else error)
        return False

    return True


def convert_data_folder(data_saved_path, output_folder,
                        eyeball_subfolder,
                        overwrite=False, verbose=True):
    """
    Convert the eyeball recordings of every participant of one study.

    Reads `<data_saved_path>/<participant>/<eyeball_subfolder>/*.webm` and
    writes `<output_folder>/<same name>.mp4`. Participants without a
    recording folder are skipped with a message.

    Args:
        data_saved_path: the configured study data root, with one sub-folder
            per participant.
        output_folder: the configured mp4 output folder read by
            `eye_detection.process_video_folder`.
        eyeball_subfolder: per-participant folder holding the recordings;
            "ET" for ImBrAid.
        overwrite: re-encode videos whose mp4 already exists (default is to
            skip them, so the call can be interrupted and resumed).
        verbose: print the progress of every file.

    Returns:
        List of the mp4 paths that were written during this call.
    """
    os.makedirs(output_folder, exist_ok=True)

    converted_paths = []
    for pid in sorted(os.listdir(data_saved_path)):
        eyeball_folder = os.path.join(data_saved_path, pid, eyeball_subfolder)
        if not os.path.exists(eyeball_folder):
            print(f"Path {eyeball_folder} does not exist, skipping participant {pid}")
            continue

        for file_name in sorted(os.listdir(eyeball_folder)):
            if not file_name.endswith('.webm'):
                continue

            webm_path = os.path.join(eyeball_folder, file_name)
            mp4_path = os.path.join(output_folder, file_name.replace('.webm', '.mp4'))
            if os.path.exists(mp4_path) and not overwrite:
                if verbose:
                    print(f"mp4 already exists for {file_name}, skipping...")
                continue

            if verbose:
                print(f"Processing {file_name} for participant {pid}...")
            if convert_webm_file(
                    webm_path, mp4_path, overwrite=True):
                converted_paths.append(mp4_path)
                if verbose:
                    print(f"Finished processing {file_name} for participant {pid}.")

    return converted_paths
