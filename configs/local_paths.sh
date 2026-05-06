#!/bin/bash

# The following lines may not be necessary, adjust them according to your setup.
ENV_NAME="ts_asr_whisper"
NFS_ARCHIVE="/data/user_data/apolok/envs/${ENV_NAME}.tar.gz"
LOCAL_SCRATCH="/scratch/apolok/${ENV_NAME}"
ACTIVATE_SCRIPT="${LOCAL_SCRATCH}/bin/activate"

# Check if environment is already extracted and ready
if [ ! -f "$ACTIVATE_SCRIPT" ]; then
    echo "Environment not found on local scratch. Extracting..."
    
    # Create local directory
    mkdir -p "$LOCAL_SCRATCH"
    
    # Extract directly from NFS to Scratch 
    tar -xzf "$NFS_ARCHIVE" -C "$LOCAL_SCRATCH"
    
    if [ $? -eq 0 ]; then
        echo "Extraction successful."
    else
        echo "Error: Extraction failed!"
        exit 1
    fi
else
    echo "Environment already present in scratch. Skipping extraction."
fi

# Source the environment
source "$ACTIVATE_SCRIPT"

# Verify activation (optional, for your logs)
echo "Using Python from: $(which python)"

# Root directory of the source code.
export SRC_ROOT=/home/apolok/CS-ASR

# Name of the Weights & Biases project.
export WANDB_PROJECT=test

# Weights & Biases entity (username or team name).
export WANDB_ENTITY=butspeechfit

# Run ID for Weights & Biases, using the EXPERIMENT variable. This variable is automatically set in the Python code, no need to change.
export WANDB_RUN_ID="${EXPERIMENT}"

# Cache directory for Hugging Face models.
export HF_HOME=/data/user_data/apolok/hf

# Set to 0 for online mode with Hugging Face Hub.
export HF_HUB_OFFLINE=0

# Add source root to the Python path.
export PYTHONPATH="$SRC_ROOT:$SRC_ROOT/src:$PYTHONPATH"

# Path for experiment outputs.
#export EXPERIMENT_PATH="/data/group_data/swl/old_home/byan/lang_diar/asr_exp2"
export EXPERIMENT_PATH="/data/user_data/apolok/CS-ASR-exp"
# Directory containing LHOTSE manifest files.
# Usually: {your_data_path}/manifests - depending on your setting in scripts/data/prepare.sh
export MANIFEST_DIR="/data/user_data/apolok/mt-asr-data/manifests/"

# Path to pretrained CTC models.
# Is used in the yaml config files. You can leave this var empty but then you must set the path in the corresponding yaml config file.
export PRETRAINED_CTC_MODELS_PATH=/data/user_data/apolok/CTC_pretrained/

# Path to the pretrained model checkpoint.
# Is used in the yaml config files. You can leave this var empty but then you must set the path in the corresponding yaml config file.
export PRETRAINED_MODEL_PATH=

# Path to musan dataset. If not used, leave empty.
export MUSAN_ROOT=/data/user_data/apolok/mt-asr-data/data/musan/musan/
