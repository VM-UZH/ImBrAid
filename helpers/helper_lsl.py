"""Select named streams from loaded XDF data."""

from helpers.project_utils.helper_utils import ET_FEATURES


def find_stream(data, name):
    candidates = [stream for stream in data if stream['info']['name'][0] == name]

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    print(f'Warning: multiple streams found for {name}, using the one with longest data')
    return max(candidates, key=lambda stream: len(stream['time_stamps']))

