#!/bin/bash
set -eo pipefail

# --- Configuration & Defaults ---
# Allow these to be overridden by environment variables, otherwise use defaults
MODEL=${MODEL:-"DiCoW_v3_3"}
ORG=${ORG:-"BUT-FIT"}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-"joint_beam_diar"}

# --- 0. Setup and Checks ---
echo ">>> [Setup] Loading configuration..."
source configs/local_paths.sh

# specific exports derived from above
export HF_MODEL_PATH="${ORG}/${MODEL}"
export EXPERIMENT="${MODEL}_${EXPERIMENT_TAG}"
CONFIG_NAME="emma_submissions/${EXPERIMENT}"
CONFIG_PATH="configs/decode/${CONFIG_NAME}.yaml"

# Validate required env vars from local_paths.sh
if [[ -z "${SRC_ROOT:-}" ]]; then
    echo "Error: SRC_ROOT is not set. Check your local_paths.sh."
    exit 1
fi

if [[ -z "${MANIFEST_DIR_DIAR:-}" ]]; then
    echo "Error: MANIFEST_DIR_DIAR is not set. Please define it in local_paths.sh."
    exit 1
fi

echo "    Model: $HF_MODEL_PATH"
echo "    Experiment: $EXPERIMENT"
echo "    Config Path: $CONFIG_PATH"


# --- 1. Diarization ---
echo ">>> [Step 1/4] Running diarization..."
# Ensure the diarization script is executable
chmod +x "$SRC_ROOT/scripts/diarize.sh"

# Run diarization
#"$SRC_ROOT/scripts/diarize.sh"


# --- 2. Create Decoding Config ---
echo ">>> [Step 2/4] Generating Hydra config at $CONFIG_PATH..."

mkdir -p "$(dirname "$CONFIG_PATH")"

cat <<'EOF' > "$CONFIG_PATH"
# @package _global_
experiment: ${oc.env:EXPERIMENT}

wandb:
  project: EMMA_leaderboard

model:
  whisper_model: ${oc.env:HF_MODEL_PATH}

data:
  use_diar: true
  dev_diar_cutsets:
    - ${oc.env:MANIFEST_DIR}/notsofar1/notsofar1_sdm_dev_set_240825.1_dev1_cutset.jsonl.gz
  eval_cutsets:
  - ${oc.env:MANIFEST_DIR}/notsofar1/notsofar1_sdm_eval_set_240629.1_eval_small_with_GT_cutset.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/ami/ami-sdm_cutset_test.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/ami/ami-ihm-mix_cutset_test.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/librimix/librimix_cutset_libri2mix_test-clean.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/librimix/librimix_cutset_libri2mix_test-clean_noisy.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/librimix/librimix_cutset_libri3mix_test-clean.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/librimix/librimix_cutset_libri3mix_test-clean_noisy.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/librispeechmix/librispeechmix_cutset_test-clean-1mix.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/librispeechmix/librispeechmix_cutset_test-clean-2mix.jsonl.gz
  - ${oc.env:MANIFEST_DIR}/librispeechmix/librispeechmix_cutset_test-clean-3mix.jsonl.gz
  eval_diar_cutsets:
    - ${oc.env:MANIFEST_DIR_DIAR}/notsofar1/notsofar1_sdm_eval_set_240629.1_eval_small_with_GT_cutset.jsonl.gz
    - ${oc.env:MANIFEST_DIR_DIAR}/ami/ami-sdm_cutset_test.jsonl.gz
    - ${oc.env:MANIFEST_DIR_DIAR}/ami/ami-ihm-mix_cutset_test.jsonl.gz
    - ${oc.env:MANIFEST_DIR_DIAR}/librimix/librimix_cutset_libri2mix_test-clean.jsonl.gz
    - ${oc.env:MANIFEST_DIR_DIAR}/librimix/librimix_cutset_libri2mix_test-clean_noisy.jsonl.gz
    - ${oc.env:MANIFEST_DIR_DIAR}/librimix/librimix_cutset_libri3mix_test-clean.jsonl.gz
    - ${oc.env:MANIFEST_DIR_DIAR}/librimix/librimix_cutset_libri3mix_test-clean_noisy.jsonl.gz
    - ${oc.env:MANIFEST_DIR_DIAR}/librispeechmix/librispeechmix_cutset_test-clean-1mix.jsonl.gz
    - ${oc.env:MANIFEST_DIR_DIAR}/librispeechmix/librispeechmix_cutset_test-clean-2mix.jsonl.gz
    - ${oc.env:MANIFEST_DIR_DIAR}/librispeechmix/librispeechmix_cutset_test-clean-3mix.jsonl.gz
  enrollment_cutsets:
    - ${oc.env:MANIFEST_DIR}/librispeech/librispeech_cutset_test-clean.jsonl.gz

training:
  decode_only: true
  eval_metrics_list: ["tcp_wer"]
  generation_num_beams: 5
  per_device_eval_batch_size: 2

decoding:
  decoding_ctc_weight: 0.2
  length_penalty: 0.1
EOF


# --- 3. Run Decoding ---
echo ">>> [Step 3/4] Running decoding..."

# We use the config we just generated.
# Note: Ensure python path is correct or activate conda env before running script
python "$SRC_ROOT/src/main.py" "+decode=${CONFIG_NAME}"


# --- 4. Create Submission File ---
echo ">>> [Step 4/4] Creating submission file..."

OUT_DIR="$SRC_ROOT/emma_hyp_files"
mkdir -p "$OUT_DIR"

python "$SRC_ROOT/utils/generate_emma_submission.py" \
  --hyp_dir="$SRC_ROOT/exp/$EXPERIMENT" \
  --out_path="$OUT_DIR/${EXPERIMENT}_hyp.json"

echo ">>> Done! Submission saved to: $OUT_DIR/${EXPERIMENT}_hyp.json"