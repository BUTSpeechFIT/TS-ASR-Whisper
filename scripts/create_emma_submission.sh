#!/bin/bash

source configs/local_paths.sh
MODEL=DiCoW_v3_2
ORG=BUT-FIT
export HF_MODEL_PATH=${ORG}/${MODEL}
CONFIG_NAME=leaderboard/sc_gt/${MODEL}_joint_beam
CONFIG_PATH=configs/decode/${CONFIG_NAME}.yaml
export EXPERIMENT="${MODEL}_joint_beam_all"

# 1. Create decoding config
mkdir -p "$(dirname "$CONFIG_PATH")"
cat <<'EOF' > "$CONFIG_PATH"
# @package _global_
experiment: ${oc.env:EXPERIMENT}

wandb:
  project: EMMA_leaderboard

model:
  whisper_model: ${oc.env:HF_MODEL_PATH}

data:
  eval_cutsets:
  - ${oc.env:MANIFEST_DIR}/notsofar1/notsofar1-sdm_cutset_eval_sc.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/ami/ami-sdm_cutset_test.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/ami/ami-ihm-mix_cutset_test.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/librimix/librimix_cutset_libri2mix_test-clean.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/librimix/librimix_cutset_libri2mix_test-clean_noisy.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/librimix/librimix_cutset_libri3mix_test-clean.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/librimix/librimix_cutset_libri3mix_test-clean_noisy.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/librispeechmix/librispeechmix_cutset_test-clean-1mix.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/librispeechmix/librispeechmix_cutset_test-clean-2mix.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/librispeechmix/librispeechmix_cutset_test-clean-3mix.jsonl.gz
  use_timestamps: true

training:
  decode_only: true
  eval_metrics_list: ["tcp_wer"]
  generation_num_beams: 5
  dataloader_num_workers: 2
  per_device_eval_batch_size: 2
  use_fddt: true

decoding:
  decoding_ctc_weight: 0.2
  condition_on_prev: false
  length_penalty: 0.1
EOF


# 2. Run decoding
$SRC_ROOT/sge_tools/interactive_python  $SRC_ROOT/src/main.py +decode=$CONFIG_NAME

# 3. Create submission file
mkdir -p $SRC_ROOT/emma_hyp_files
$SRC_ROOT/sge_tools/interactive_python $SRC_ROOT/utils_scripts/generate_emma_submission.py \
 --hyp_dir=$SRC_ROOT/exp/$EXPERIMENT --out_path=$SRC_ROOT/emma_hyp_files/${EXPERIMENT}_hyp.json



