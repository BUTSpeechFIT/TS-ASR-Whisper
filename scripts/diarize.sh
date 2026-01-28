#!/bin/bash

# Exit on error (-e), or pipe failures (-o pipefail)
set -eo pipefail

# ==============================================================================
# Setup & Configuration
# ==============================================================================

# Activate your environment
CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate diarizen

export EXPERIMENT="DiariZen"

# Load environment configuration
CONFIG_PATH="configs/local_paths.sh"
if [[ -f "$CONFIG_PATH" ]]; then
    source "$CONFIG_PATH"
else
    echo "Error: Configuration file not found at $CONFIG_PATH"
    exit 1
fi

MODEL_ID="BUT-FIT/diarizen-wavlm-large-s80-md"
EXP_ROOT="${SRC_ROOT}/diar_exp/${EXPERIMENT}"
COLLAR_VALUE=0.25

# ==============================================================================
# Cutsets Definition (Tuple format: "CUTSET_NAME USE_ALIGNMENT")
# USE_ALIGNMENT: true = use align_and_compute_der, false = use compute_der
# ==============================================================================
CUTSETS=(
    "notsofar1/notsofar1_sdm_eval_set_240629.1_eval_small_with_GT_cutset false"
    "ami/ami-sdm_cutset_test false"
    "ami/ami-ihm-mix_cutset_test false"
    "librimix/librimix_cutset_libri2mix_test-clean true"
    "librimix/librimix_cutset_libri2mix_test-clean_noisy true"
    "librimix/librimix_cutset_libri3mix_test-clean true"
    "librimix/librimix_cutset_libri3mix_test-clean_noisy true"
    "librispeechmix/librispeechmix_cutset_test-clean-1mix true"
    "librispeechmix/librispeechmix_cutset_test-clean-2mix true"
    "librispeechmix/librispeechmix_cutset_test-clean-3mix true"
)

mkdir -p "$EXP_ROOT"

# ==============================================================================
# Helper Functions
# ==============================================================================

log() {
    echo -e "[$(date +'%H:%M:%S')] $1"
}

run_pipeline() {
    # 1. Parse the "tuple" (Name + Flag)
    local entry="$1"
    local cutset_name
    local use_align

    # Split string by space into array variables
    read -r cutset_name use_align <<< "$entry"

    local input_manifest="${MANIFEST_DIR}/${cutset_name}.jsonl.gz"
    local output_dir="${EXP_ROOT}/${cutset_name}"
    local output_manifest="${EXP_ROOT}/${cutset_name}.jsonl.gz"
    local output_manifest_aligned="${EXP_ROOT}/${cutset_name}_aligned.jsonl.gz"

    log "---------------------------------------------------"
    log "Processing: $cutset_name"
    log "Alignment Enabled: $use_align"
    log "---------------------------------------------------"

    if [[ ! -f "$input_manifest" ]]; then
        log "Error: Input manifest not found at $input_manifest"
        return 1
    fi

    if [[ -f "$output_manifest" ]]; then
        log "Skipping inference: Output already exists at $output_manifest"
    else
        log "Step 1/2: Running Diarization Inference..."
        mkdir -p "$output_dir"

        python "$SRC_ROOT/utils/diarizen_diar.py" \
            --model="$MODEL_ID" \
            --input_cutset="$input_manifest" \
            --output_dir="$output_dir"

        log "Step 2/2: Converting RTTM to Lhotse Cutset..."
        python "$SRC_ROOT/utils/prepare_diar_cutset_from_rttm_dir.py" \
            --lhotse_manifest_path="$input_manifest" \
            --rttm_dir="$output_dir" \
            --out_manifest_path="$output_manifest"
    fi

    log "Computing DER (Collar: ${COLLAR_VALUE})..."

    if [[ "$use_align" == "true" ]]; then
        # NOTE: We need it for se-dicow if we want to generate other speakers mixture
        # on the fly so that we can find target speaker cuts from enrollment recordings set.
        log "Mode: Align and Compute DER"
        python "$SRC_ROOT/utils/align_and_compute_der_between_cutsets.py" \
            --ref_cutset="$input_manifest" \
            --hyp_cutset="$output_manifest" \
            --hyp_cutset_out="$output_manifest_aligned" \
            --collar="$COLLAR_VALUE"
    else
        log "Mode: Standard Compute DER"
        python "$SRC_ROOT/utils/compute_der_between_cutsets.py" \
            --ref_cutset="$input_manifest" \
            --hyp_cutset="$output_manifest" \
            --collar="$COLLAR_VALUE"
    fi

    log "Done with $cutset_name"
}

# ==============================================================================
# Main Execution
# ==============================================================================

for ENTRY in "${CUTSETS[@]}"; do
    run_pipeline "$ENTRY"
done

log "All cutsets processed successfully."