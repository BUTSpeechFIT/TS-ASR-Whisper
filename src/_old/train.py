import os
from functools import reduce

import lhotse
from safetensors.torch import load_file
from transformers import HfArgumentParser
from transformers.utils import logging

from data.local_datasets import TS_ASR_Dataset, TS_ASR_Random_Dataset, DataCollator, get_text_norm
from models.containers import WhisperQKContainer, WhisperContainer, get_optimizer
from utils.evaluation import compute_metrics
from utils.trainers import CustomTrainer
from utils.training_args import ModelArguments, DataArguments, DecodingArguments, CustomTrainingArguments

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments, DecodingArguments, CustomTrainingArguments))
    model_args, data_args, decoding_args, training_args = parser.parse_args_into_dataclasses()

    # 1. Load the training data
    train_cutsets = reduce(lambda a, b: a + b, [lhotse.load_manifest(cutset) for cutset in data_args.train_cutsets])
    eval_cutsets = reduce(lambda a, b: a + b, [lhotse.load_manifest(cutset) for cutset in data_args.eval_cutsets])

    # 2. Create dataset instances
    text_norm = get_text_norm(data_args.eval_text_norm)
    train_dataset_class = TS_ASR_Random_Dataset if data_args.use_random_segmentation else TS_ASR_Dataset
    train_dataset = train_dataset_class(train_cutsets, do_augment=data_args.do_augment,
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
                                        max_r_crop=data_args.max_r_crop)

    eval_dataset = TS_ASR_Dataset(eval_cutsets,
                                  text_norm=text_norm,
                                  use_timestamps=data_args.use_timestamps,
                                  audio_path_prefix=data_args.audio_path_prefix,
                                  audio_path_prefix_replacement=data_args.audio_path_prefix_replacement,
                                  )

    dev_decoding_dataset = eval_dataset.get_subset(
        data_args.dev_decoding_samples) if data_args.dev_decoding_samples else eval_dataset

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
                              target_amp_use_non_target=model_args.target_amp_use_non_target)

    # 4. Get the model and possibly load pretrained weights
    model = container.model

    if model_args.reinit_encoder_from:
        model.get_encoder().load_state_dict(load_file(model_args.reinit_encoder_from))
    elif model_args.reinit_from:
        state_dict = load_file(model_args.reinit_from)
        state_dict['proj_out.weight'] = state_dict['model.decoder.embed_tokens.weight']
        model.load_state_dict(state_dict)

    # 5. Initialize trainer
    collator = DataCollator(feature_extractor=container.feature_extractor, tokenizer=container.tokenizer,
                            bos_token_id=container.model.config.decoder_start_token_id,
                            max_length=training_args.generation_max_length)

    trainer = CustomTrainer(model=model, args=training_args,
                            eval_dataset=dev_decoding_dataset if training_args.predict_with_generate else eval_dataset,
                            data_collator=collator,
                            train_dataset=train_dataset, tokenizer=container.tokenizer, container=container,
                            optimizers=(get_optimizer(model, training_args), None)
                            )

    # 6. Apply custom metric computation if needed
    if training_args.predict_with_generate:
        model.generation_config.ctc_weight = decoding_args.decoding_ctc_weight


        def _compute_metrics(pred):
            step = trainer.state.global_step
            current_dir = f'{training_args.output_dir}/dev/{step}'
            os.makedirs(current_dir, exist_ok=True)
            return compute_metrics(pred=pred,
                                   output_dir=current_dir,
                                   text_norm=text_norm,
                                   tokenizer=container.tokenizer,
                                   decode_with_timestamps=data_args.use_timestamps
                                   )


        trainer.compute_metrics = _compute_metrics

    # 7. Train the model
    if not training_args.decode_only:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # 8. Evaluate the model
    trainer.args.predict_with_generate = True
    if decoding_args.decoding_ctc_weight is not None:
        model.generation_config.ctc_weight = decoding_args.decoding_ctc_weight
    outputs = trainer.predict(test_dataset=eval_dataset)

    if trainer.accelerator.is_main_process:
        logger.info(f"Test loss: {outputs.metrics}")
        logger.info(
            f"Metrics {compute_metrics(output_dir=trainer.args.output_dir, text_norm=text_norm, tokenizer=container.tokenizer, pred=outputs)}")
