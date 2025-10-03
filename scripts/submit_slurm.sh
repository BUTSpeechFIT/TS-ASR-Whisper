#!/bin/bash
#SBATCH --job-name DiCoW
#SBATCH --partition qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node 8
#SBATCH --cpus-per-task=64
#SBATCH --time 48:00:00
#SBATCH --output=logs/%x_%j.out

if [ $# -eq 0 ]; then
  echo "No extra config provided, using default config."
fi

source $SRC_ROOT/configs/local_paths.sh

cd $SRC_ROOT || exit;conda activate ts_asr_whisper

export OMP_NUM_THREADS=32
torchrun --standalone --nnodes=1 --nproc-per-node=8 "${SRC_ROOT}/src/main.py" "$@"