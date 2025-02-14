#!/bin/bash

unset PYTHONPATH
unset PYTHONHOME
conda activate ts_asr_whisper  # Activate the freshly-created Conda environment.

# Root directory of the source code.
export SRC_ROOT=

# Name of the Weights & Biases project.
export WANDB_PROJECT=

# Weights & Biases entity (username or team name).
export WANDB_ENTITY=

# Run ID for Weights & Biases, using the EXPERIMENT variable. This variable is automatically set in the Python code, no need to change.
export WANDB_RUN_ID="${EXPERIMENT}"

# Cache directory for Hugging Face models.
export HF_HOME=

# Set to 0 for online mode with Hugging Face Hub.
export HF_HUB_OFFLINE=0

# Add source root to the Python path.
export PYTHONPATH="$SRC_ROOT:$PYTHONPATH"

# Add SCTK binaries to the system path.
export PATH="/path/to/SCTK/bin:$PATH"

# Path for experiment outputs.
export EXPERIMENT_PATH="${SRC_ROOT}/exp/${EXPERIMENT}"

# Directory containing manifest files.
export MANIFEST_DIR=

# Prefix in audio paths to be replaced.
export AUDIO_PATH_PREFIX=

# Replacement prefix for audio paths.
export AUDIO_PATH_PREFIX_REPLACEMENT=

# Path to cached LibriSpeech training data - HuggingFace.
export LIBRI_TRAIN_CACHED_PATH=

# Path to cached LibriSpeech development data - HuggingFace.
export LIBRI_DEV_CACHED_PATH=

# Path to pretrained CTC models.
export PRETRAINED_CTC_MODELS_PATH=

# Path to the pretrained model checkpoint.
export PRETRAINED_MODEL_PATH=
