import argparse
from json import dumps

from lhotse import load_manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-cutset', type=str, required=True)
    parser.add_argument('--output-json', type=str, required=True)

    args = parser.parse_args()

    cut_set = load_manifest(args.input_cutset)
    res_json = []
    for r in cut_set:
        for sup in r.supervisions:
            res_json.append({
                'start_time': sup.start,
                'end_time': sup.end,
                'session_id': r.id,
                'speaker': sup.speaker,
                'words': 'DUMMY WORDS',
            })

    with open(args.output_json, 'w') as f:
        f.write(dumps(res_json, indent=4))
