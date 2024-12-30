import argparse

from lhotse import load_manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_manifest', type=str, required=True)
    parser.add_argument('--output_manifest', type=str, required=True)
    parser.add_argument('--max_len', type=int, default=30)  # seconds

    args = parser.parse_args()
    
    cuts = load_manifest(args.input_manifest)
    cuts = cuts.filter(lambda cut: cut.duration <= args.max_len)
    cuts.to_file(args.output_manifest)
