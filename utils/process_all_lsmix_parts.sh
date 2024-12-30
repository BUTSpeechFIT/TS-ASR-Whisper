#!/bin/bash

set -eou pipefail

SUPSET_PATH="/export/fs06/dklemen1/k2_icefall/icefall/egs/libricss/SURT/data/manifests"
WAV_PATH="/export/fs06/dklemen1/chime_followup/data/librimix/storage"
OUT_CUTSET_PATH="/export/fs06/dklemen1/chime_followup/data/librimix/LibriMix/lhotse_recipes"

mix_types=(
    Libri2Mix
    Libri3Mix
)

N_PARTS=4
parts=(
    "dev"
    "test"
    "train-100"
    "train-360"
)
cutsets=(
    "librispeech_supervisions_dev-clean.jsonl.gz"
    "librispeech_supervisions_test-clean.jsonl.gz"
    "librispeech_supervisions_train-clean-100.jsonl.gz"
    "librispeech_supervisions_train-clean-360.jsonl.gz"
)
types=(
    "both"
    "clean"
)

for mix_type in "${mix_types[@]}"; do
    for type in "${types[@]}"; do
        for part in $(seq 0 $((N_PARTS-1))); do
            supset_path=$SUPSET_PATH/${cutsets[$part]}
            wav_dir_path=$WAV_PATH/$mix_type/wav16k/max/${parts[$part]}/mix_${type}
            out_path=$OUT_CUTSET_PATH/${cutsets[$part]}_${mix_type}_${type}_cutset_sc.json
            
            python utils/lsmix_to_lhotse.py \
                --ls_supset=$supset_path \
                --mixture_wavs_dir=$wav_dir_path \
                --output_manifest=$out_path
        done
    done
done
