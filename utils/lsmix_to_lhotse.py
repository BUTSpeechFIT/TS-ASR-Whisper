import argparse
import os

from lhotse import CutSet, MonoCut, fix_manifests, load_manifest
from lhotse.audio import Recording
from tqdm import tqdm

def main(ls_supset, mixture_wavs_dir, output_manifest):
    mixed_cuts = []
    for wav in tqdm(os.listdir(mixture_wavs_dir)):
        if '.wav' not in wav:
            continue

        utt_id = wav.split('.')[0]
        source_ids = utt_id.split('_')

        rec = Recording.from_file(f'{mixture_wavs_dir}/{wav}', utt_id)
        sups = []

        for src in source_ids:
            if src not in ls_supset:
                print(f'{src} not found in LS cuts')
                continue
            
            sups.append(ls_supset[src])

        for sup in sups:
            sup.recording_id = rec.id

        mixed_cuts.append(MonoCut(
            id=utt_id,
            start=0,
            duration=rec.duration,
            channel=0,
            recording=rec,
            supervisions=sups
        ))

    CutSet.from_cuts(mixed_cuts).to_json(output_manifest)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ls_supset', type=str, required=True)
    parser.add_argument('--mixture_wavs_dir', type=str, required=True)
    parser.add_argument('--output_manifest', type=str, required=True)

    args = parser.parse_args()
    
    main(load_manifest(args.ls_supset), args.mixture_wavs_dir, args.output_manifest)
