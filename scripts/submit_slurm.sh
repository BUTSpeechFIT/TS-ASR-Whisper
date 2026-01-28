#!/bin/bash
#SBATCH --job-name DiCoW
#SBATCH --partition qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node 8
#SBATCH --cpus-per-task=128
#SBATCH --time 48:00:00
#SBATCH --output=logs/%x_%j.out

if [ $# -eq 0 ]; then
  echo "No extra config provided, using default config."
fi

source $SRC_ROOT/configs/local_paths.sh

cd $SRC_ROOT || exit

export OMP_NUM_THREADS=16

# Load Anaconda
ml Anaconda3/2024.02-1

# Initialize conda in non-interactive shell
source $(conda info --base)/etc/profile.d/conda.sh

# Activate your environment
conda activate ts_asr

# Load FFmpeg module
ml FFmpeg/6.0-GCCcore-13.2.0

torchrun --standalone --nnodes=1 --nproc-per-node=8 "${SRC_ROOT}/src/main.py" "$@"
