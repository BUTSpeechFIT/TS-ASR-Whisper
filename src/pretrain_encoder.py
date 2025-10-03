import os

from safetensors.torch import load_file
from transformers.utils import logging

from data.collators import DataCollatorForPretraining
from data.local_datasets import get_libri_dataset
from models.containers import WhisperContainer, get_optimizer
from txt_norm import get_text_norm
from utils.decoding import ctc_greedy_decode
from utils.evaluation import compute_metrics
from utils.trainers import CustomTrainerEncoder
from utils.training_args import Cfg

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


def main(cfg: Cfg):
    model_args, data_args, decoding_args, training_args = cfg.model, cfg.data, cfg.decoding, cfg.training

    text_norm = get_text_norm(data_args.train_text_norm)

    # 3. Initialize container class
    container = WhisperContainer(model_type=model_args.whisper_model, pretrained_encoder=model_args.pretrained_encoder,
                                 ctc_weight=model_args.ctc_weight,
                                 training_args=training_args, predict_timestamps=data_args.use_timestamps,
                                 fddt_is_diagonal=model_args.fddt_is_diagonal,
                                 fddt_bias_only=model_args.fddt_bias_only,
                                 fddt_use_silence=model_args.fddt_use_silence,
                                 fddt_use_target=model_args.fddt_use_target,
                                 fddt_use_overlap=model_args.fddt_use_overlap,
                                 fddt_use_non_target=model_args.fddt_use_non_target,
                                 remove_timestamps_from_ctc=training_args.remove_timestamps_from_ctc,
                                 apply_fddt_to_n_layers=model_args.apply_fddt_to_n_layers,
                                 use_fddt=training_args.use_fddt,
                                 fddt_init=model_args.fddt_init,
                                 non_target_fddt_value=model_args.non_target_fddt_value,
                                 mt_num_speakers=model_args.mt_num_speakers if model_args.mt_asr else 1,
                                 params_to_keep_frozen_keywords=model_args.params_to_keep_frozen_keywords,
                                 use_initial_fddt=model_args.use_initial_fddt,
                                 global_lang_id=data_args.global_lang_id,
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

    train_dset, eval_dset = get_libri_dataset(text_norm)

    collator = DataCollatorForPretraining(feature_extractor=container.feature_extractor, tokenizer=container.tokenizer,
                                          bos_token_id=container.model.config.decoder_start_token_id,
                                          max_length=training_args.generation_max_length,
                                          use_timestamps=data_args.use_timestamps)

    trainer_enc = CustomTrainerEncoder(model=encoder,
                                       args=training_args,
                                       train_dataset=train_dset,
                                       eval_dataset=eval_dset,
                                       data_collator=collator,
                                       preprocess_logits_for_metrics=(
                                           lambda predictions, labels: ctc_greedy_decode(
                                               predictions, len(container.tokenizer.get_vocab()),
                                               model.config.pad_token_id
                                           )),
                                       optimizers=(get_optimizer(model, training_args), None),
                                       tokenizer=container.tokenizer, container=container)

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
