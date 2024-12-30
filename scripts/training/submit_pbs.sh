#!/bin/bash
#PBS -N wh
#PBS -q gpu_dgx@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=70:mem=256gb:ngpus=4:mpiprocs=1
#PBS -l walltime=10:00:00
#PBS -o /storage/brno12-cerit/home/dklement/speech/chime_followup/CHIME2024/exp/log_out/$PBS_JOBNAME_$PBS_JOBID.out
#PBS -e /storage/brno12-cerit/home/dklement/speech/chime_followup/CHIME2024/exp/log_out/$PBS_JOBNAME_$PBS_JOBID.err

export N_GPUS=4
export CUDA_VISIBLE_DEVICES="0,1,2,3"

/storage/brno12-cerit/home/dklement/speech/chime_followup/CHIME2024/scripts/training/run_train.sh "$CFG_PATH"
