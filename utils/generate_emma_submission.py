import argparse
import os
from pathlib import Path
import meeteval


def main(hyp_dir, out_path, step):
    # iterate over different directories
    hyps = []
    for dataset in os.listdir(hyp_dir):
        dataset_path = Path(hyp_dir) / dataset
        if not dataset_path.is_dir():
            continue

        # Iterate over each reference file in the task directory
        for hyp_file in dataset_path.glob(f"{step}/wer/*/tcp_wer_hyp.json"):
            hyps.append(meeteval.io.load(hyp_file))
    hyps = meeteval.io.SegLST.merge(*hyps)
    hyps.dump(out_path)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp_dir', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--step', default="0", type=str)
    return parser


if __name__ == '__main__':
    parser = argparser()
    args = parser.parse_args()
    main(**vars(args))
