#!/bin/bash
API_KEY=0rJnU9JCQ3Qc3ERUqgkvDx83jO9VeSIT
PYTHON=/mnt/matylda5/ipoloka/miniconda3/envs/tsasr/bin/python
PROCESS_SCRIPT=/mnt/matylda5/ipoloka/projects/TS-ASR-Whisper/process_cutsets_voxtral.py
SCORE_SCRIPT=/mnt/matylda5/ipoloka/projects/TS-ASR-Whisper/score_voxtral.py
OUTPUT_DIR=voxtral_output

cutsets=(
    /mnt/scratch/tmp/ipoloka/mt_asr_data/manifests/notsofar1/notsofar1_sdm_eval_set_240629.1_eval_small_with_GT_cutset.jsonl.gz
    /mnt/scratch/tmp/ipoloka/mt_asr_data/manifests/ami/ami-sdm_cutset_test.jsonl.gz
    /mnt/scratch/tmp/ipoloka/mt_asr_data/manifests/librispeechmix/librispeechmix_cutset_test-clean-1mix.jsonl.gz
    /mnt/scratch/tmp/ipoloka/mt_asr_data/manifests/librispeechmix/librispeechmix_cutset_test-clean-2mix.jsonl.gz
    /mnt/scratch/tmp/ipoloka/mt_asr_data/manifests/librispeechmix/librispeechmix_cutset_test-clean-3mix.jsonl.gz
    /mnt/scratch/tmp/ipoloka/mt_asr_data/manifests/mixer6/mixer6-sdm_cutset_eval.jsonl.gz
     /mnt/scratch/tmp/ipoloka/mt_asr_data/manifests/dipco/dipco-sdm_cutset_eval.jsonl.gz
)

for cutset in "${cutsets[@]}"; do
    stem=$(basename "$cutset" .jsonl.gz)
    results_json="${OUTPUT_DIR}/${stem}_voxtral_results.json"

    echo "========================================================"
    echo "Processing: $stem"
    echo "========================================================"

    $PYTHON $PROCESS_SCRIPT \
        --api-key "$API_KEY" \
        --cutset-paths "$cutset" \
        --output-dir "$OUTPUT_DIR"

    if [ $? -ne 0 ]; then
        echo "ERROR: processing failed for $stem, skipping scoring."
        continue
    fi

    echo "Scoring: $stem"

    $PYTHON $SCORE_SCRIPT \
        --reference-cutset "$cutset" \
        --voxtral-results "$results_json" \
        --output-dir "$OUTPUT_DIR" \
        --text-norm

    if [ $? -ne 0 ]; then
        echo "ERROR: scoring failed for $stem"
    fi
done
