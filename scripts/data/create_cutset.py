import argparse

from lhotse import load_manifest, fix_manifests, CutSet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_recset', type=str, required=True, help='Path to the recordings manifest')
    parser.add_argument('--input_supset', type=str, required=True, help='Path to the supervisions manifest')
    parser.add_argument('--output', type=str, required=True, help='Path to the output manifest')

    args = parser.parse_args()

    rc = load_manifest(args.input_recset)
    ss = load_manifest(args.input_supset)

    CutSet.from_manifests(*fix_manifests(rc, ss)).to_file(args.output)
