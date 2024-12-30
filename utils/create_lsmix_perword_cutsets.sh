#!/bin/bash

lsmix_path="/export/fs06/dklemen1/chime_followup/data/librimix/LibriMix/lhotse_recipes"

for fn in $(ls $lsmix_path); do
    if echo $fn | grep "train" > /dev/null; then
        false
    else
        continue
    fi

    echo "Processing $fn"

    res_fn="${fn%.*}_wlevel.jsonl.gz"
    python create_perword_cutset_from_alignment_sups.py \
            --input_cutset_path="${lsmix_path}/${fn}" \
            --output_cutset_path="${lsmix_path}/${res_fn}"
        
done
