from models.dicow.modeling_dicow import DiCoWForConditionalGeneration
from models.dicow.config import DiCoWConfig
from huggingface_hub import HfApi
import os
from transformers import AutoTokenizer, AutoFeatureExtractor

def argparse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pretrained model checkpoint')
    parser.add_argument('--model_name', type=str, default='SE_DiCoW', help='Name of the model')
    parser.add_argument('--org', type=str, default='BUT-FIT', help='Hugging Face organization name')
    parser.add_argument('--base_whisper_model', type=str, default='openai/whisper-large-v3-turbo',)

    return parser.parse_args()

if __name__ == '__main__':

    args = argparse()
    NEW_MODEL_ID = f"{args.org}/{args.model_name}"
    DiCoWConfig.register_for_auto_class()
    DiCoWForConditionalGeneration.register_for_auto_class("AutoModelForSpeechSeq2Seq")
    api = HfApi()

    dicow_pretrained = DiCoWForConditionalGeneration.from_pretrained(args.model_path)
    dicow_config = DiCoWConfig(**dicow_pretrained.config.to_dict())
    dicow_model = DiCoWForConditionalGeneration(dicow_config)
    dicow_model.load_state_dict(dicow_pretrained.state_dict())
    fe = AutoFeatureExtractor.from_pretrained(args.base_whisper_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_whisper_model)

    dicow_model.push_to_hub(NEW_MODEL_ID)
    fe.push_to_hub(NEW_MODEL_ID)
    tokenizer.push_to_hub(NEW_MODEL_ID)
    api.upload_file(
        path_or_fileobj=f"{os.environ['SRC_ROOT']}/export_sources/readmes/{args.model_name}.md",
        path_in_repo="README.md",
        repo_id=NEW_MODEL_ID,
    )
    api.upload_file(
        path_or_fileobj=f"{os.environ['SRC_ROOT']}/export_sources/images/{args.model_name}.png",
        path_in_repo=f"{args.model_name}.png",
        repo_id=NEW_MODEL_ID,
    )
    api.upload_file(
        path_or_fileobj=f"{os.environ['SRC_ROOT']}/export_sources/generation_config.json",
        path_in_repo="generation_config.json",
        repo_id=NEW_MODEL_ID,
    )

