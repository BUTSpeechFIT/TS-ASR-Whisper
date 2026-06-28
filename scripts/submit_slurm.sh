#!/bin/bash
#SBATCH --job-name CS
#SBATCH --partition general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time 48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mem=256G
mkdir -p logs

if [ $# -eq 0 ]; then
  echo "No extra config provided, using default config."
fi


cd $SRC_ROOT || exit

source configs/local_paths.sh

export OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1

torchrun --standalone --nnodes=1 --nproc-per-node=4 "${SRC_ROOT}/src/main.py" "$@"
