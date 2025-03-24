#!/bin/bash

# The following lines may not be necessary, adjust them according to your setup.
unset PYTHONPATH
unset PYTHONHOME
source /path/to/.bashrc  # Source the user's bash configuration
conda activate ts_asr_whisper  # Activate the freshly-created Conda environment.

# Root directory of the source code.
export SRC_ROOT=

# Name of the Weights & Biases project.
export WANDB_PROJECT=

# Weights & Biases entity (username or team name).
export WANDB_ENTITY=

# Run ID for Weights & Biases, using the EXPERIMENT variable. This variable is automatically set in the Python code, no need to change.
export WANDB_RUN_ID="${EXPERIMENT}"
# Used to download NOTSOFAR dataset.
export HF_TOKEN=""

# Cache directory for Hugging Face models.
export HF_HOME=

# Set to 0 for online mode with Hugging Face Hub.
export HF_HUB_OFFLINE=0

# Add source root to the Python path.
export PYTHONPATH="$SRC_ROOT:$PYTHONPATH"

# (Deprecated) Add SCTK binaries to the system path - we changed the scoring script to meeteval, so you can leave this one intact!
# Will remove in the future!
export PATH="/path/to/SCTK/bin:$PATH"

# Path for experiment outputs.
export EXPERIMENT_PATH="${SRC_ROOT}/exp/${EXPERIMENT}"

# Directory containing LHOTSE manifest files.
# Usually: {your_data_path}/manifests - depending on your setting in scripts/data/prepare.sh
export MANIFEST_DIR="${SRC_ROOT}/data/manifests"

# (OPTIONAL) Prefix in audio paths to be replaced.
# Is used when u copy audio data to SSD to speed up the training but you don't want to create new manifests
export AUDIO_PATH_PREFIX=

# (OPTIONAL) Replacement prefix for audio paths.
# Is related to the parameter above.
export AUDIO_PATH_PREFIX_REPLACEMENT=

# (OPTIONAL) Path to cached LibriSpeech training data - HuggingFace.
export LIBRI_TRAIN_CACHED_PATH=

# (OPTIONAL) Path to cached LibriSpeech development data - HuggingFace.
export LIBRI_DEV_CACHED_PATH=

# Path to pretrained CTC models.
# Is used in the yaml config files. You can leave this var empty but then you must set the path in the corresponding yaml config file.
export PRETRAINED_CTC_MODELS_PATH=

# Path to the pretrained model checkpoint.
# Is used in the yaml config files. You can leave this var empty but then you must set the path in the corresponding yaml config file.
export PRETRAINED_MODEL_PATH=
