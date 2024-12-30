#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48           ### Number of Tasks per CPU
#SBATCH --mem=256G                    ### Memory required, 4 gigabyte
#SBATCH --account=a100acct          ### Account name
#SBATCH --partition=gpu-a100         ### Cheaha Partition
#SBATCH --time=30:00:00             ### Estimated Time of Completion, 16 hour
#SBATCH --output=/export/fs06/dklemen1/chime_followup/CHIME2024_new/exp_icassp/out_logs/%x_%j.out          ### Slurm Output file, %x is job name, %j is job id
#SBATCH --error=/export/fs06/dklemen1/chime_followup/CHIME2024_new/exp_icassp/out_logs/%x_%j.err           ### Slurm Error file, %x is job name, %j is job id\

export N_GPUS=4
export CUDA_VISIBLE_DEVICES="$(seq -s ',' 0 $((N_GPUS-1)))"

/export/fs06/dklemen1/chime_followup/CHIME2024_new/scripts/training/run_train.sh "$@"
