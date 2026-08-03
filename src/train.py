import copy
import gc
import os
from functools import reduce
from typing import Dict, Any

import lhotse
import torch
from safetensors.torch import load_file
from transformers import EarlyStoppingCallback
from transformers.trainer_pt_utils import get_model_param_count
from transformers.utils import logging

from data.collators import DataCollator
from data.local_datasets import build_datasets, TS_ASR_Dataset, load_cutsets, LhotseLongFormDataset
from peft.utils import ModulesToSaveWrapper

from models.containers import WhisperContainer, get_optimizer
from txt_norm import get_text_norm
from utils.evaluation import compute_longform_metrics
from utils.general import create_lower_uppercase_mapping, patch_wandb_init_with_config, update_generation_config
from utils.trainers import CustomTrainer, GradLogger
from utils.training_args import Cfg

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


class ModelTrainer:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.model_args = cfg.model
        self.data_args = cfg.data
        self.decoding_args = cfg.decoding
        self.training_args = cfg.training
        self.aug_args = cfg.aug

        self.container = None
        self.model = None
        self.trainer = None
        self.text_norm = None

    def _initialize_container(self):
        """Initialize the model container with appropriate configuration."""
        self.container = WhisperContainer(
            model_args=self.model_args,
            data_args=self.data_args,
            remove_timestamps_from_ctc=self.training_args.remove_timestamps_from_ctc,
            use_fddt=self.training_args.use_fddt,
            use_lora=self.training_args.use_lora,
            params_to_keep_frozen_keywords=self.model_args.params_to_keep_frozen_keywords,
        )

    def _load_training_cutsets(self):
        """Load and prepare training cutsets.

        Lazily: training corpora are read on demand from disk (via LazyCutReader) instead of
        being fully materialized into one big in-memory Cut graph, which is what DataLoader
        workers were duplicating (CPython refcounting defeats fork's copy-on-write sharing on
        large Python object graphs -- see the more_data worker-RSS investigation).
        """
        train_cutsets = load_cutsets(self.data_args.train_cutsets, self.data_args.use_enrollments,
                                     use_prev_prompt=self.data_args.use_prev_prompt, lazy=True)
        return train_cutsets

    def _create_enrollment_cutset(self):
        """Create enrollment cutset if needed."""
        if (self.data_args.use_enrollments and
                self.data_args.enrollment_cutsets is not None):
            return reduce(lambda x, y: x + y,
                          [lhotse.load_manifest(cutset) for cutset in self.data_args.enrollment_cutsets])
        return None

    def _create_train_dataset(self, train_cutsets, enrollment_cutset):
        """Create training dataset."""
        train_dataset = TS_ASR_Dataset(
            train_cutsets,
            do_augment=self.aug_args.do_augment,
            dataset_weights=self.data_args.dataset_weights,
            use_timestamps=self.data_args.use_timestamps,
            use_prev_prompt=self.data_args.use_prev_prompt,
            musan_root=self.aug_args.musan_root,
            musan_augment_prob=self.aug_args.musan_augment_prob,
            text_norm=get_text_norm(self.data_args.train_text_norm),
            feature_extractor=self.container.feature_extractor,
            global_lang_id=self.data_args.global_lang_id,
            load_channel_zero_only=self.data_args.load_channel_zero_only,
            use_enrollments=self.data_args.use_enrollments,
            enrollment_cutset=enrollment_cutset,
            num_other_speakers=self.data_args.number_of_mixed_speakers,
            min_overlap_ratio=self.data_args.min_enrollment_mix_overlap,
            max_overlap_ratio=self.data_args.max_enrollment_mix_overlap,
        )

        return train_dataset

    def _create_eval_datasets(self, enrollment_cutset):
        """Create development and evaluation datasets."""
        dev_datasets = build_datasets(
            self.data_args.dev_cutsets, self.data_args,
            self.text_norm, self.container, self.data_args.dev_diar_cutsets,
            enrollment_cutset=enrollment_cutset,
            dataset_class=LhotseLongFormDataset
        )

        eval_datasets = build_datasets(
            self.data_args.eval_cutsets, self.data_args,
            self.text_norm, self.container, self.data_args.eval_diar_cutsets,
            enrollment_cutset=enrollment_cutset,
            dataset_class=LhotseLongFormDataset
        )

        return dev_datasets, eval_datasets

    def _load_model_weights(self):
        """Load pretrained model weights if specified."""
        if self.model_args.reinit_encoder_from:
            enc_state_dict = load_file(self.model_args.reinit_encoder_from)
            enc_state_dict_no_fddt = {k: v for k, v in enc_state_dict.items() if 'fddt' not in k}
            encoder = self.model.get_encoder()
            if isinstance(encoder, ModulesToSaveWrapper):
                encoder = encoder.modules_to_save[encoder.active_adapters[0]]
            logger.info(encoder.load_state_dict(enc_state_dict_no_fddt, strict=False))

        if self.model_args.reinit_from:
            state_dict = self._load_state_dict(self.model_args.reinit_from)
            state_dict['proj_out.weight'] = state_dict['model.decoder.embed_tokens.weight']
            logger.info(f'Loading model weights from: {self.model_args.reinit_from}')
            logger.info(self.model.load_state_dict(state_dict, strict=False))

    def _load_state_dict(self, path: str) -> Dict[str, Any]:
        """Load state dictionary from file or directory."""
        if path.endswith('.safetensors'):
            return load_file(path)

        # Load all safetensors files in directory and merge
        state_dict = {}
        for file in os.listdir(path):
            if file.endswith('.safetensors'):
                state_dict.update(load_file(os.path.join(path, file)))
        return state_dict

    def _log_model_parameters(self):
        """Log FDDT and SCB parameters."""
        fddts = [n for n, _ in self.model.named_parameters() if 'fddt' in n]
        logger.info(f"FDDTs: {fddts}")

    def _create_data_collator(self):
        """Create appropriate data collator."""

        return DataCollator(
            feature_extractor=self.container.feature_extractor,
            tokenizer=self.container.tokenizer,
            bos_token_id=self.container.model.config.decoder_start_token_id,
            max_length=self.training_args.generation_max_length,
            stno_gaussian_noise_var=self.aug_args.stno_gaussian_noise_var,
            stno_gaussian_noise_prob=self.aug_args.stno_gaussian_noise_prob,
            stno_segment_augment_prob=self.aug_args.stno_segment_augment_prob,
            stno_segment_change_prob=self.aug_args.stno_segment_change_prob,
            stno_min_segment_length=self.aug_args.stno_min_segment_length,
            stno_max_segment_length=self.aug_args.stno_max_segment_length,
            spec_aug_prob=self.aug_args.spec_aug_prob,
            use_enrollments=self.data_args.use_enrollments,
            use_prev_prompt=self.data_args.use_prev_prompt,
        )

    def _create_compute_metrics_fn(self, dev_datasets):
        """Create metrics computation function."""

        def _compute_metrics(pred, dset=None, split='dev', metrics_list=None):
            step = self.trainer.state.global_step
            output_dir = f'{self.trainer.args.output_dir}/{split}/{step}'
            os.makedirs(output_dir, exist_ok=True)
            return compute_longform_metrics(
                pred, self.trainer, output_dir, self.text_norm,
                self.training_args.train_metrics_list if metrics_list is None else metrics_list,
                dset,
                save_visualizations=self.training_args.save_visualizations,
            )

        return _compute_metrics

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        if "wandb" in self.training_args.report_to:
            patch_wandb_init_with_config(self.cfg, self.training_args.store_src)

            if self.training_args.watch_grads and self.trainer.accelerator.is_main_process:
                self.trainer.add_callback(GradLogger(self.model))

    def _preheat_enabled(self):
        """Whether an FDDT-only preheat phase should be run before the main training."""
        if (self.training_args.use_fddt_only_n_epochs <= 0 and
                self.training_args.use_fddt_only_n_steps <= 0):
            return False
        if not self.model_args.prefixes_to_preheat:
            logger.warning("use_fddt_only_n_steps/epochs is set but prefixes_to_preheat is empty, "
                           "skipping the preheat phase.")
            return False
        if self.training_args.resume_from_checkpoint:
            # A checkpoint can only come from the main phase (the preheat phase writes none),
            # so its weights already contain the preheated FDDTs.
            logger.info(f"Resuming from {self.training_args.resume_from_checkpoint}, skipping the preheat phase.")
            return False
        return True

    def _preheat_training_args(self):
        """Training arguments for the preheat run, derived from the main ones.

        Shallow copy on purpose: `TrainingArguments` carries an already-initialised
        `distributed_state`, which must be shared rather than deep-copied. Any field
        overridden below is rebound, never mutated in place.
        """
        args = copy.copy(self.training_args)
        args.output_dir = os.path.join(self.training_args.output_dir, "preheat")
        args.run_name = f"{self.training_args.run_name}_preheat"

        if self.training_args.use_fddt_only_n_steps > 0:
            args.max_steps = self.training_args.use_fddt_only_n_steps
        else:
            args.max_steps = -1
            args.num_train_epochs = self.training_args.use_fddt_only_n_epochs

        args.eval_strategy = "no"
        args.save_strategy = "no"
        args.load_best_model_at_end = False
        args.predict_with_generate = False
        args.report_to = []
        return args

    def _run_preheat_phase(self, train_dataset, collator):
        """Train only `prefixes_to_preheat` (FDDTs & friends) as a separate training run.

        Everything is then thrown away except the model weights: the main training that follows
        builds a brand new trainer, optimizer and LR scheduler (so the warmup is spent on the
        full model, not on the preheat). Continuing the preheat optimizer/schedule into the
        unfrozen phase -- what the previous in-training-loop unfreezing did -- dropped the whole
        model in at peak LR and made the loss explode.
        """
        if not self._preheat_enabled():
            return

        args = self._preheat_training_args()
        self.container.freeze_except(self.model_args.prefixes_to_preheat)
        duration = (f"{args.max_steps} steps" if args.max_steps > 0 else f"{args.num_train_epochs} epochs")
        logger.info(f"***** Preheating {self.model_args.prefixes_to_preheat} for {duration} *****")
        logger.info(f"  Number of trainable parameters = "
                    f"{get_model_param_count(self.model, trainable_only=True):,}")

        preheat_trainer = CustomTrainer(
            model=self.model,
            args=args,
            data_collator=collator,
            train_dataset=train_dataset,
            processing_class=self.container.tokenizer,
            container=self.container,
            optimizers=(get_optimizer(self.model, args, self.model_args.prefixes_to_preheat), None),
            params_to_keep_frozen=self.model_args.params_to_keep_frozen_keywords,
        )
        preheat_trainer.train()
        # Preheating can take a while, so keep its result around: if the main phase dies before
        # its first checkpoint, this can be fed back through model.reinit_from.
        preheat_trainer.save_model(args.output_dir)
        logger.info(f"***** Preheat finished, weights saved to {args.output_dir} *****")

        # Give up every reference to the preheat run (optimizer states, DDP/accelerate wrappers,
        # dataloaders) so the main phase starts from a clean slate. `self.model` is untouched.
        preheat_trainer.accelerator.free_memory()
        del preheat_trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Keep the overall step budget unchanged: max_steps counted the preheat steps before.
        if self.training_args.max_steps > 0 and self.training_args.use_fddt_only_n_steps > 0:
            self.training_args.max_steps -= self.training_args.use_fddt_only_n_steps
            logger.info(f"Main training phase will run for {self.training_args.max_steps} steps.")

        self.container.freeze_by_keywords(self.model_args.params_to_keep_frozen_keywords)
        logger.info(f"***** Unfreezing params except {self.model_args.params_to_keep_frozen_keywords} *****")
        logger.info(f"  Number of trainable parameters = "
                    f"{get_model_param_count(self.model, trainable_only=True):,}")

    def do_eval(self, eval_datasets, decoding_ctc_weight, eval_metrics_list, condition_key):
        """Perform evaluation on given datasets."""
        _compute_metrics = self._create_compute_metrics_fn(eval_datasets)

        # Update compute_metrics for trainer
        self.trainer.compute_metrics = (
            lambda x: _compute_metrics(
                x, eval_datasets[self.trainer.metric_key_prefix.removeprefix(f"{condition_key}_")],
                split=self.trainer.metric_key_prefix, metrics_list=eval_metrics_list
            )
        )

        # Perform evaluation
        self.trainer.args.predict_with_generate = True
        if decoding_ctc_weight is not None:
            self.model.generation_config.ctc_weight = decoding_ctc_weight

        metrics = self.trainer.evaluate(eval_dataset=eval_datasets, metric_key_prefix=condition_key)
        logger.info(f"Metrics {metrics}")

    def train(self):
        """Main training pipeline."""
        logger.info(f"Config: {self.cfg}")

        # Initialize components
        self._initialize_container()
        self.text_norm = get_text_norm(self.data_args.eval_text_norm)

        # Load data
        train_cutsets = self._load_training_cutsets()
        enrollment_cutset = self._create_enrollment_cutset()
        train_dataset = self._create_train_dataset(train_cutsets, enrollment_cutset)
        dev_datasets, eval_datasets = self._create_eval_datasets(enrollment_cutset)

        # Setup model
        self.model = self.container.model
        create_lower_uppercase_mapping(self.container.tokenizer)
        self._log_model_parameters()
        self._load_model_weights()
        update_generation_config(self.model, self.training_args, self.decoding_args,
                                 predict_timestamps=self.data_args.use_timestamps)

        collator = self._create_data_collator()

        # FDDT-only preheat, run to completion before anything else is built: the main trainer
        # below must see the final (unfrozen) requires_grad layout when it builds its optimizer.
        if not self.training_args.decode_only:
            self._run_preheat_phase(train_dataset, collator)

        # Create trainer
        callbacks = ([EarlyStoppingCallback(self.training_args.early_stopping_patience)]
                     if self.training_args.early_stopping_patience > 0 else None)

        self.trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            eval_dataset=dev_datasets,
            data_collator=collator,
            train_dataset=train_dataset,
            processing_class=self.container.tokenizer,
            container=self.container,
            optimizers=(get_optimizer(self.model, self.training_args, self.model_args.prefixes_to_preheat), None),
            callbacks=callbacks,
            params_to_keep_frozen=self.model_args.params_to_keep_frozen_keywords,
        )

        # Setup additional components
        self._setup_wandb()

        # Setup metrics computation
        if self.training_args.predict_with_generate:
            self.model.generation_config.ctc_weight = self.decoding_args.decoding_ctc_weight
            _compute_metrics = self._create_compute_metrics_fn(dev_datasets)

            self.trainer.compute_metrics = (
                lambda x: _compute_metrics(
                    x, dev_datasets[self.trainer.metric_key_prefix.removeprefix("eval_")],
                    split=self.trainer.metric_key_prefix,
                    metrics_list=self.training_args.train_metrics_list
                )
            )

        # Train and evaluate
        if not self.training_args.decode_only:
            self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)

        self.do_eval(eval_datasets, self.decoding_args.decoding_ctc_weight,
                     self.training_args.eval_metrics_list, "test")


def main(cfg: Cfg) -> None:
    """Main entry point for training."""
    trainer = ModelTrainer(cfg)
    trainer.train()
