import os

import lhotse
import wandb
from safetensors.torch import load_file
from transformers import EarlyStoppingCallback
from transformers.utils import logging

from data.local_datasets import build_dataset, TS_ASR_Dataset, TS_ASR_Random_Dataset, DataCollator, get_text_norm
from models.containers import WhisperQKContainer, WhisperContainer, get_optimizer
from mt_asr.dataset import MT_ASR_Dataset, MT_Data_Collator
from utils.evaluation import compute_longform_metrics
from utils.general import create_lower_uppercase_mapping
from utils.generation import update_generation_config
from utils.trainers import CustomTrainer
from utils.training_args import Cfg

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


def main(cfg: Cfg) -> None:
    logger.info(f"Config: {cfg}")
    model_args, data_args, decoding_args, training_args = cfg.model, cfg.data, cfg.decoding, cfg.training
    # 1. Load the training data
    train_cutsets = [lhotse.load_manifest(cutset) for cutset in data_args.train_cutsets]

    # 2. Create dataset instances
    text_norm = get_text_norm(data_args.eval_text_norm)
    train_dataset_class = TS_ASR_Random_Dataset if data_args.use_random_segmentation else TS_ASR_Dataset
    train_dataset = train_dataset_class(train_cutsets, do_augment=data_args.do_augment,
                                        dataset_weights=data_args.dataset_weights,
                                        use_timestamps=data_args.use_timestamps,
                                        musan_noises=data_args.musan_noises,
                                        text_norm=get_text_norm(data_args.train_text_norm),
                                        empty_transcript_ratio=data_args.empty_transcripts_ratio,
                                        train_with_diar_outputs=data_args.train_with_diar_outputs,
                                        audio_path_prefix=data_args.audio_path_prefix,
                                        audio_path_prefix_replacement=data_args.audio_path_prefix_replacement,
                                        vad_from_alignments=data_args.vad_from_alignments,
                                        random_sentence_l_crop_p=data_args.random_sentence_l_crop_p,
                                        random_sentence_r_crop_p=data_args.random_sentence_r_crop_p,
                                        max_l_crop=data_args.max_l_crop,
                                        max_r_crop=data_args.max_r_crop,
                                        )

    # 3. Initialize container class
    container_cls = WhisperQKContainer if model_args.use_qk_biasing else WhisperContainer
    container = container_cls(model_type=model_args.whisper_model, pretrained_encoder=model_args.pretrained_encoder,
                              ctc_weight=model_args.ctc_weight, shift_pos_embeds=model_args.shift_pos_embeds,
                              training_args=training_args, predict_timestamps=data_args.use_timestamps,
                              target_amp_is_diagonal=model_args.target_amp_is_diagonal,
                              target_amp_bias_only=model_args.target_amp_bias_only,
                              target_amp_use_silence=model_args.target_amp_use_silence,
                              target_amp_use_target=model_args.target_amp_use_target,
                              target_amp_use_overlap=model_args.target_amp_use_overlap,
                              target_amp_use_non_target=model_args.target_amp_use_non_target,
                              remove_timestamps_from_ctc=training_args.remove_timestamps_from_ctc,
                              apply_target_amp_to_n_layers=model_args.apply_target_amp_to_n_layers,
                              use_target_amplifiers=training_args.use_target_amplifiers,
                              target_amp_init=model_args.target_amp_init,
                              mt_num_speakers=model_args.mt_num_speakers if model_args.mt_asr else 1,
                              )

    # Create mapping between lower case and upper case tokens
    create_lower_uppercase_mapping(container.tokenizer)

    dev_dataset = build_dataset(data_args.dev_cutsets, data_args, decoding_args, text_norm, container,
                                data_args.dev_diar_cutsets)
    eval_dataset = build_dataset(data_args.eval_cutsets, data_args, decoding_args, text_norm, container,
                                 data_args.eval_diar_cutsets)
    if model_args.mt_asr:
        train_dataset = MT_ASR_Dataset(train_dataset, model_args.mt_num_speakers)
        dev_dataset = MT_ASR_Dataset(dev_dataset, model_args.mt_num_speakers)
        eval_dataset = MT_ASR_Dataset(eval_dataset, model_args.mt_num_speakers)

    # 4. Get the model, possibly load pretrained weights and update generation config
    model = container.model

    target_amplifiers = [n for n, _ in model.named_parameters() if 'target_amplifiers' in n]
    logger.info(f"Target amplifiers: {target_amplifiers}")

    if model_args.reinit_encoder_from:
        enc_state_dict = load_file(model_args.reinit_encoder_from)
        enc_state_dict_no_amplifiers = {k: v for k, v in enc_state_dict.items() if 'target_amplifiers' not in k}
        logger.info(model.get_encoder().load_state_dict(enc_state_dict_no_amplifiers, strict=False))

    if model_args.reinit_from:
        if model_args.reinit_from.endswith('.safetensors'):
            state_dict = load_file(model_args.reinit_from)
        else:
            # load all safetensors files in directory and merge to single dictionary
            state_dict = {}
            for file in os.listdir(model_args.reinit_from):
                if file.endswith('.safetensors'):
                    state_dict.update(load_file(os.path.join(model_args.reinit_from, file)))
        state_dict['proj_out.weight'] = state_dict['model.decoder.embed_tokens.weight']
        logger.info('Loading model weights from: ' + model_args.reinit_from)
        logger.info(model.load_state_dict(state_dict, strict=False))

    update_generation_config(model, training_args, decoding_args)

    # 5. Initialize trainer
    collator_class = MT_Data_Collator if model_args.mt_asr else DataCollator
    collator = collator_class(feature_extractor=container.feature_extractor, tokenizer=container.tokenizer,
                              bos_token_id=container.model.config.decoder_start_token_id,
                              mask_inputs=data_args.mask_inputs,
                              max_length=training_args.generation_max_length)

    trainer = CustomTrainer(model=model, args=training_args,
                            eval_dataset=dev_dataset,
                            data_collator=collator,
                            train_dataset=train_dataset, tokenizer=container.tokenizer, container=container,
                            optimizers=(get_optimizer(model, training_args, model_args.prefixes_to_preheat), None),
                            callbacks=[EarlyStoppingCallback(
                                training_args.early_stopping_patience)] if training_args.early_stopping_patience > 0 else None,
                            params_to_keep_frozen=model_args.params_to_keep_frozen_keywords,
                            )

    if training_args.use_amplifiers_only_n_epochs > 0 or training_args.use_amplifiers_only_n_steps > 0:
        container.model.freeze_except(model_args.prefixes_to_preheat)

    if not model_args.reinit_from:
        container.model.suppress_interactions()

    # 6. Apply custom metric computation if needed
    if training_args.predict_with_generate:
        model.generation_config.ctc_weight = decoding_args.decoding_ctc_weight

        def _compute_metrics(pred, dset=None, split='dev', metrics_list=None):
            step = trainer.state.global_step
            output_dir = f'{trainer.args.output_dir}/{split}/{step}'
            os.makedirs(output_dir, exist_ok=True)
            return compute_longform_metrics(pred, trainer, output_dir, text_norm,
                                            training_args.train_metrics_list if metrics_list is None else metrics_list,
                                            dset)

        trainer.compute_metrics = (lambda x: _compute_metrics(x, dev_dataset))

    # 7. Train the model
    if not training_args.decode_only:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # If we don't pass an extra dataset, compute_longform_metrics would use the trainer.eval_dataset == dev_dataset.
    trainer.compute_metrics = (
        lambda x: _compute_metrics(x, eval_dataset, split='test', metrics_list=training_args.eval_metrics_list))

    # 8. Evaluate the model
    trainer.args.predict_with_generate = True
    if decoding_args.decoding_ctc_weight is not None:
        model.generation_config.ctc_weight = decoding_args.decoding_ctc_weight
    outputs = trainer.predict(test_dataset=eval_dataset)
    metrics = outputs.metrics
    logger.info(f"Metrics {metrics}")
    if wandb.run is not None:
        wandb.log({f"test/{key}": val for key, val in metrics.items()})
