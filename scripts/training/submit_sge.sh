#!/bin/bash
#$ -N CHIME
#$ -cwd
#$ -v SRC_ROOT
#$ -q long.q
#$ -l ram_free=180G,mem_free=180G
#$ -l scratch=4
#$ -l gpu=4,gpu_ram=20G
#$ -o /mnt/scratch/tmp/$USER/CHiME8-NOTSOFAR1/exp/log/$JOB_NAME_$JOB_ID.out
#$ -e /mnt/scratch/tmp/$USER/CHiME8-NOTSOFAR1/exp/log/$JOB_NAME_$JOB_ID.err

set -eux
# As Karel said don't be an idiot and use the same number of GPUs as requested
export N_GPUS=4
export $(gpus $N_GPUS) || exit 1

# e.g. submit_sge.sh "+decode=mt_asr/mt_nsf ++training.per_device_eval_batch_size=1"
CFG="$1"
[ -z "$1" ] && CFG="$CFG_PATH"
[ -z "$SRC_ROOT" ] && { echo "Please export SRC_ROOT=path/to/CHIME2024"; exit 1; }

cd $SRC_ROOT
./scripts/training/run_train.sh "$CFG"
