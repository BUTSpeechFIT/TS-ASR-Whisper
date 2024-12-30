#!/usr/bin/bash
#$ -N CHIME_infer
#$ -q long.q
#$ -l ram_free=40G,mem_free=40G
#$ -l scratch=1
#$ -t 1-18:1
#$ -l gpu=1,gpu_ram=20G
#$ -o /mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/outputs/$JOB_NAME_$JOB_ID_$TASK_ID.out
#$ -e /mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/outputs/$JOB_NAME_$JOB_ID_$TASK_ID.err

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

# Get start index by multiplying the task id by the 10
start_index=$((($SGE_TASK_ID - 1) * 10))
end_index=$((($SGE_TASK_ID) * 10))

# Those defined in https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments + custom
args=(
--output_dir "/mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/outputs/eval_decoding_system_gt2222222222222222222222"
--whisper_model="openai/whisper-large-v3"
--generation_num_beams=1
--decoding_ctc_weight=0
--generation_max_length=225
--init_from="/mnt/scratch/tmp/ipoloka/experiments/Whisper_large_dev_v4/checkpoint-650"
#--init_from="/mnt/scratch/tmp/ipoloka/experiments/timestamps_WhisperCTC_large_resume_v2_2/checkpoint-2500"
--start_decoding_index=$start_index
--end_decoding_index=$end_index
#--use_soft_diar_labels
--use_gt_diar
)

"${SRC_ROOT}/sge_tools/python" "${SRC_ROOT}/infer_eval.py" "${args[@]}"

