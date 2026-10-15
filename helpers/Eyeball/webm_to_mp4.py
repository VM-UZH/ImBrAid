"""Convert Eyeball WebM recordings to H.264 MP4 files.

Dependencies: source WebM files, ffmpeg-python and the FFmpeg executable."""

import os

import ffmpeg

VIDEO_CODEC = 'libx264'
CRF = 15
PRESET = 'fast'


def convert_webm_file(webm_path, mp4_path, overwrite=False):
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
