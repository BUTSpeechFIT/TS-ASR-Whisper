#!/usr/bin/bash
#$ -N CHIME_infer
#$ -q long.q
#$ -l ram_free=40G,mem_free=40G
#$ -l scratch=1
#$ -l gpu=1,gpu_ram=20G
#$ -o /mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/outputs/$JOB_NAME_$JOB_ID.out
#$ -e /mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/outputs/$JOB_NAME_$JOB_ID.err


# As Karel said don't be an idiot and use the same number of GPUs as requested
export N_GPUS=1

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger files
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited


SRC_ROOT="/mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge"

export HF_HOME="/mnt/scratch/tmp/ipoloka/hf_cache"

# Those defined in https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments + custom
args=(
--output_dir "${SRC_ROOT}/outputs/decoding_gt_timestamps_fix"
--whisper_model="openai/whisper-small.en"
--per_device_eval_batch_size=1
--generation_num_beams=1
#--decoding_ctc_weight=0.3
--per_device_eval_batch_size=16
--generation_max_length=225
--init_from="/mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge_exp/experiments/WhisperCTC_5e_0.3ctc_lr1e4_spk_embed_v4_diag_w_timestamps/checkpoint-2407/model.safetensors"
--use_gt_diar
)

"${SRC_ROOT}/sge_tools/python" "${SRC_ROOT}/infer.py" "${args[@]}"
