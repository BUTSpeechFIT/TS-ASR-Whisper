#!/bin/bash

source ../configs/local_paths.sh
export PYTHONPATH="$(dirname ${BASH_SOURCE[0]})/../:$PYTHONPATH"

# SRC_ROOT is defined in local_paths.sh
DATA_DIR= $SRC_ROOT/data
MANIFESTS_DIR=$DATA_DIR/manifests

mkdir -p $DATA_DIR
mkdir -p $MANIFESTS_DIR
mkdir -p $DATA_DIR/tmp

# Download lhotse manifests

# LS
librispeech_dir=$DATA_DIR/librispeech/LibriSpeech
lhotse download librispeech $DATA_DIR/librispeech
lhotse prepare librispeech $librispeech_dir $MANIFESTS_DIR

# AMI
lhotse download ami --mic sdm $DATA_DIR/ami
lhotse prepare ami --mic sdm $DATA_DIR/ami $MANIFESTS_DIR

# Rename AMI manifests to match the naming convention
for sfx in "supervisions" "recordings"; do
    for partition in "train" "dev" "test"; do
        mv $MANIFESTS_DIR/ami-sdm_${sfx}_${partition}.jsonl.gz $MANIFESTS_DIR/ami-sdm_${partition}_sc_${sfx}.jsonl.gz
    done
done

git clone https://github.com/JorisCos/LibriMix $DATA_DIR/tmp/LibriMix
pip install -r $DATA_DIR/tmp/LibriMix/requirements.txt

# Download WHAM
wham_zip_file=$DATA_DIR/tmp/wham/wham_noise.zip
wham_folder=$DATA_DIR/tmp/wham/wham_noise
if [ ! -d "$wham_folder" ]; then
    mkdir -p $DATA_DIR/tmp/wham

    if [ ! -f "$wham_zip_file" ]; then
        wget -c --tries=0 --read-timeout=20 https://storage.googleapis.com/whisper-public/wham_noise.zip -P $DATA_DIR/tmp/wham
    fi

    unzip -qn $DATA_DIR/tmp/wham/wham_noise.zip -d $DATA_DIR/tmp/wham
    rm -rf $DATA_DIR/tmp/wham/wham_noise.zip
fi

metadata_dir=$DATA_DIR/tmp/LibriMix/metadata/Libri2Mix
python $DATA_DIR/tmp/LibriMix/scripts/augment_train_noise.py --wham_dir $DATA_DIR/tmp/wham/wham_noise
python $DATA_DIR/tmp/LibriMix/scripts/create_librimix_from_metadata.py --librispeech_dir $librispeech_dir \
    --wham_dir $DATA_DIR/tmp/wham/wham_noise \
    --metadata_dir $metadata_dir \
    --librimix_outdir $DATA_DIR/libri2mix \
    --n_src 2 \
    --freqs 16k \
    --modes max \
    --types mix_clean mix_both mix_single

# Prepare L2Mix manifests
for n_hours in 100; do
    for type in "clean" "both"; do
        python $DATA_DIR/lsmix_to_lhotse.py --ls_supset $MANIFESTS_DIR/librispeech_supervisions_train-clean-$n_hours.jsonl.gz \
            --mixture_wavs_dir $DATA_DIR/libri2mix/Libri2Mix/wav16k/max/train-$n_hours/mix_$type \
            --output_manifest $MANIFESTS_DIR/libri2mix_${type}_${n_hours}_train_sc_cutset.jsonl.gz
        python $DATA_DIR/filter_long_cuts.py --input $MANIFESTS_DIR/libri2mix_${type}_${n_hours}_train_sc_cutset.jsonl.gz --output $MANIFESTS_DIR/libri2mix_${type}_${n_hours}_train_sc_cutset_30s.jsonl.gz --max_len 30
    done
done

for partition in "dev" "test"; do
    for type in "clean" "both"; do
        python $DATA_DIR/lsmix_to_lhotse.py --ls_supset $MANIFESTS_DIR/librispeech_supervisions_$partition-clean.jsonl.gz \
            --mixture_wavs_dir $DATA_DIR/libri2mix/Libri2Mix/wav16k/max/$partition/mix_$type \
            --output_manifest $MANIFESTS_DIR/libri2mix_mix_${type}_sc_${partition}_cutset.jsonl.gz
    done
done

# Setup NSF
az storage copy --recursive --destination $DATA_DIR/tmp/nsf --source https://notsofarsa.blob.core.windows.net/benchmark-datasets --include-path train_set/240825.1_train/MTG
az storage copy --recursive --destination $DATA_DIR/tmp/nsf --source https://notsofarsa.blob.core.windows.net/benchmark-datasets --include-path dev_set/240825.1_dev1/MTG
az storage copy --recursive --destination $DATA_DIR/tmp/nsf --source https://notsofarsa.blob.core.windows.net/benchmark-datasets --include-path eval_set/240629.1_eval_small_with_GT/MTG

mkdir -p $DATA_DIR/nsf
mv $DATA_DIR/tmp/nsf/benchmark-datasets/* $DATA_DIR/nsf/

# Setup NSF Manifests
python $DATA_DIR/nsf_to_lhotse.py \
    --dataset_path $DATA_DIR/nsf/train_set/240825.1_train/MTG \
    --output_dir $MANIFESTS_DIR \
    --output_fname_prefix "notsofar_train_sc_cutset"

python $DATA_DIR/nsf_to_lhotse.py \
    --dataset_path $DATA_DIR/nsf/dev_set/240825.1_dev1/MTG \
    --output_dir $MANIFESTS_DIR \
    --output_fname_prefix "notsofar_dev_sc_cutset"

python $DATA_DIR/nsf_to_lhotse.py \
    --dataset_path $DATA_DIR/nsf/eval_set/240629.1_eval_small_with_GT/MTG \
    --output_dir $MANIFESTS_DIR \
    --output_fname_prefix "notsofar_eval_sc_cutset"

# Prepare cutsets
for rec_manifest in $(ls $MANIFESTS_DIR/*recordings* ); do
    sup_manifest=${rec_manifest//recordings/supervisions}
    cset=${rec_manifest//recordings/cutset}
    python $DATA_DIR/create_cutset.py --input_recset $rec_manifest --input_supset $sup_manifest --output $cset
done

# Segment manifests to 30s chunks
python $DATA_DIR/pre_segment_using_alignments.py --input $MANIFESTS_DIR/notsofar_train_sc_cutset.jsonl.gz --output $MANIFESTS_DIR/notsofar_train_sc_cutset_30s.jsonl.gz --max_len 30
python $DATA_DIR/pre_segment_using_alignments.py --input $MANIFESTS_DIR/ami-sdm_train_sc_cutset.jsonl.gz --output $MANIFESTS_DIR/ami-sdm_train_sc_cutset_30s.jsonl.gz --max_len 30

# Prepare joint AMI+NSF dev and eval cutsets. They are used for model validation/testing during training on all of the datasets.
lhotse combine $MANIFESTS_DIR/ami-sdm_dev_sc_cutset.jsonl.gz $MANIFESTS_DIR/notsofar_dev_sc_cutset.jsonl.gz $MANIFESTS_DIR/ami_notsofar_dev_sc_cutset.jsonl.gz
lhotse combine $MANIFESTS_DIR/ami-sdm_test_sc_cutset.jsonl.gz $MANIFESTS_DIR/notsofar_eval_sc_cutset.jsonl.gz $MANIFESTS_DIR/ami_notsofar_eval_sc_cutset.jsonl.gz

# Remove tmp files, prepare lhotse manifests.
rm -rf $DATA_DIR/tmp
