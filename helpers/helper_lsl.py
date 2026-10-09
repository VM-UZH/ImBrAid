from helpers.project_utils.helper_utils import ET_FEATURES


def find_stream(data, name):
    candidates = {}
    for i, stream in enumerate(data):
        if stream['info']['name'][0] == name:
            candidates[i] = stream

    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates[list(candidates.keys())[0]]
    if len(candidates) > 1:
        print(f'Warning: multiple streams found for {name}, using the one with longest data')
        return candidates[max(candidates, key=lambda x: len(candidates[x]['time_series']))]

