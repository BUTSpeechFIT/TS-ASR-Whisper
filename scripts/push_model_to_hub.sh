source configs/local_paths.sh
python $SRC_ROOT/utils/export_dicow.py --model_path=$SRC_ROOT/new_models/se_dicow --model_name=SE-DiCoW --org BUT-FIT --base_whisper_model openai/whisper-large-v3-turbo
