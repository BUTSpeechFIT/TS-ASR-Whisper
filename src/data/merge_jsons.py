import os
import argparse

from json import loads, dumps


def load_jsons(dir_path):
    for f in os.listdir(dir_path):
        if f.endswith('.json'):
            with open(os.path.join(dir_path, f), 'r') as f:
                yield loads(f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-dir', type=str, required=True)
    parser.add_argument('--out-json-path', type=str, required=True)

    res = []

    for json_obj in load_jsons(parser.parse_args().json_dir):
        res.extend(json_obj)

    with open(parser.parse_args().out_json_path, 'w') as f:
        f.write(dumps(res))