import os

import lhotse
from safetensors.torch import load_file
from transformers.utils import logging

from data.collators import DataCollatorForPretraining
from data.local_datasets import TS_ASR_Dataset, build_datasets
from models.containers import WhisperContainer, get_optimizer
from txt_norm import get_text_norm
from utils.decoding import ctc_greedy_decode
from utils.evaluation import compute_metrics
from utils.trainers import CustomTrainerEncoder
from utils.training_args import Cfg

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


def main(cfg: Cfg):
    model_args, data_args, aug_args, decoding_args, training_args = cfg.model, cfg.data, cfg.aug, cfg.decoding, cfg.training

    text_norm = get_text_norm(data_args.train_text_norm)

    # 3. Initialize container class
    container = WhisperContainer(
        model_args=model_args,
        data_args=data_args,
        use_flash_attention=training_args.use_flash_attention,
        remove_timestamps_from_ctc=training_args.remove_timestamps_from_ctc,
        use_fddt=training_args.use_fddt,
        params_to_keep_frozen_keywords=model_args.params_to_keep_frozen_keywords,
    )

    # 4. Get the model and possibly load pretrained weights
    model = container.model
    encoder = model.get_encoder()

    if model_args.reinit_encoder_from:
        logger.info(encoder.load_state_dict(load_file(model_args.reinit_encoder_from)))

    prefixes_to_disable = ['additional_layer',
                           'additional_self_attention_layer',
                           'lm_head',
                           'subsample_conv1',
                           'subsample_conv2']
    for name, param in encoder.named_parameters():
        param.requires_grad = False
        for prefix in prefixes_to_disable:
            if name.startswith(prefix):
                param.requires_grad = True

    train_cutsets = [lhotse.load_manifest(cutset) for cutset in data_args.train_cutsets]

    train_dataset = TS_ASR_Dataset(
        train_cutsets,
        dataset_weights=data_args.dataset_weights,
        use_timestamps=data_args.use_timestamps,
        text_norm=get_text_norm(data_args.train_text_norm),
        feature_extractor=container.feature_extractor,
        global_lang_id=data_args.global_lang_id,
    )

    dev_datasets = build_datasets(
        data_args.dev_cutsets, data_args,
        text_norm, container, use_ids_as_transcripts=False)

    eval_datasets = build_datasets(
        data_args.eval_cutsets, data_args,
        text_norm, container, data_args.eval_diar_cutsets, use_ids_as_transcripts=False)

    collator = DataCollatorForPretraining(feature_extractor=container.feature_extractor, tokenizer=container.tokenizer,
                                          bos_token_id=container.model.config.decoder_start_token_id,
                                          max_length=training_args.generation_max_length,
                                          use_timestamps=data_args.use_timestamps)

    trainer_enc = CustomTrainerEncoder(model=encoder,
                                       args=training_args,
                                       train_dataset=train_dataset,
                                       eval_dataset=dev_datasets,
                                       data_collator=collator,
                                       preprocess_logits_for_metrics=(
                                           lambda predictions, labels: ctc_greedy_decode(
                                               predictions, len(container.tokenizer.get_vocab()),
                                               model.config.pad_token_id
                                           )),
                                       optimizers=(get_optimizer(model, training_args), None),
                                       processing_class=container.tokenizer, container=container)

    def _compute_metrics(pred):
        step = trainer_enc.state.global_step
        current_dir = f'{training_args.output_dir}/dev/{step}'
        os.makedirs(current_dir, exist_ok=True)
        return compute_metrics(pred=pred,
                               output_dir=current_dir,
                               text_norm=text_norm,
                               tokenizer=container.tokenizer,
                               )

    trainer_enc.compute_metrics = _compute_metrics
    trainer_enc.train()
    trainer_enc.evaluate(eval_datasets)
