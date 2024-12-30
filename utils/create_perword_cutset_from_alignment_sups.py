"""
    Creates a per-word cutset from the SupervisionSegments containing w-level alignments.
"""

import argparse

from lhotse import CutSet, load_manifest, fastcopy
from tqdm import tqdm

def main(cset, output_cset_path):
    wlevel_cset = []
    for r in tqdm(cset):
        all_correct = True
        for sup in r.supervisions:
            if not sup.alignment:
                all_correct = False
                break
        
        if not all_correct:
            print('skipping', r.id)
            continue
        
        for sup in r.supervisions:    
            for i, alig in enumerate(sup.alignment['word']):
                assert alig.start >= 0
                assert alig.duration >= 0
                wlevel_cset.append(fastcopy(
                    r,
                    id=f'{r.id}_{sup.id}_{i}',
                    start=alig.start,
                    duration=alig.duration,
                    supervisions=[
                        fastcopy(
                            sup,
                            id=f'{sup.id}_{i}',
                            start=0,
                            duration=alig.duration,
                            text=alig.symbol,
                            alignment=None
                        )
                    ]
                ))

    CutSet.from_cuts(wlevel_cset).to_file(output_cset_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_cutset_path', type=str, required=True)
    parser.add_argument('--output_cutset_path', type=str, required=True)

    args = parser.parse_args()

    cset = load_manifest(args.input_cutset_path)
    main(cset, args.output_cutset_path)
