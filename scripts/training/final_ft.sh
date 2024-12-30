#!/usr/bin/bash
#$ -N CHIME
#$ -q long.q
#$ -l ram_free=180G,mem_free=180G
#$ -l scratch=8
#$ -l h=supergpu10|supergpu14
#$ -l gpu=3,gpu_ram=40G
#$ -o /mnt/scratch/tmp/ipoloka/outputs/$JOB_NAME_$JOB_ID.out
#$ -e /mnt/scratch/tmp/ipoloka/outputs/$JOB_NAME_$JOB_ID.err


# As Karel said don't be an idiot and use the same number of GPUs as requested
export N_GPUS=3

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger files
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited


SRC_ROOT="/mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge"

EXPERIMENT="Whisper_large_dev_v4"

export WANDB_PROJECT="whisper_finetune"
export WANDB_ENTITY="butspeechfit"
export WANDB_RUN_ID="${EXPERIMENT}"
export HF_HOME="/mnt/scratch/tmp/ipoloka/hf_cache"

EXPERIMENT_PATH="/mnt/scratch/tmp/ipoloka/experiments/${EXPERIMENT}"
# Those defined in https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments + custom
args=(
--output_dir="${EXPERIMENT_PATH}"
--whisper_model="openai/whisper-large-v3"
--remove_unused_columns False
--logging_steps=2
--generation_num_beams=1
--data_prefix
"/mnt/scratch/tmp/xkleme15/asr/chime/chime8_dasr_v3/notsofar1_v2/lhotse/segmented_gt_alig"
--decoding_ctc_weight=0
--do_train
--load_best_model_at_end
--remove_unused_columns False
--per_device_train_batch_size=2
--gradient_accumulation_steps=32
--logging_steps=1
--dataloader_num_workers=1
--per_device_eval_batch_size=4
--generation_max_length=225
--num_train_epochs=5
--eval_strategy="epoch"
--save_strategy="epoch"
--ctc_weight="0.3"
--learning_rate="7e-7"
--warmup_steps=0
--use_timestamps
--predict_with_generate
--dev_decoding_samples=200
--reinit_from="/mnt/scratch/tmp/ipoloka/experiments/timestamps_WhisperCTC_large_v4.1/checkpoint-6000/"
--ddp_find_unused_parameters=False
)

"${SRC_ROOT}/sge_tools/python" "${SRC_ROOT}/ft.py" "${args[@]}"

