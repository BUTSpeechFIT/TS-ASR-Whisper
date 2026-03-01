import os
from functools import reduce
from typing import Dict, Any

import lhotse
import torch
from peft.utils.save_and_load import _insert_adapter_name_into_state_dict
from safetensors.torch import load_file
from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.utils import logging

from data.local_datasets import build_datasets, load_cutsets
from models.dixtral.collator import DataCollator
from models.dixtral.container import DixtralContainer
from models.dixtral.dataset import TS_ASR_Dataset_ as TS_ASR_Dataset, LhotseLongFormDataset_ as LhotseLongFormDataset
from txt_norm import get_text_norm
from utils.evaluation import compute_longform_metrics
from utils.general import patch_wandb_init_with_config, update_generation_config
from utils.trainers import CustomTrainer, GradLogger
from utils.training_args import Cfg

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


class SaveNonPeftParamsCallback(TrainerCallback):
    """Save only non-PEFT (base model / custom) parameters, ignoring PEFT adapter weights."""

    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir,
            f"checkpoint-{state.global_step}",
        )
        os.makedirs(checkpoint_folder, exist_ok=True)

        model = kwargs["model"]

        # Unwrap from DeepSpeed / FSDP / DDP if needed
        if hasattr(model, "module"):
            model = model.module

        # Get all PEFT parameter names to exclude
        peft_param_names = set(model.peft_model.state_dict().keys()) if hasattr(model, "peft_model") \
            else {name for name, _ in model.named_parameters() if "lora_" in name or "adapter_" in name}

        # Strip the "base_model.model." prefix that LoRA adds, to match the
        # key format expected by _load_model_weights when use_lora=True
        # (it re-adds that prefix itself before calling load_state_dict)
        lora_prefix = "base_model.model."
        non_peft_state_dict = {}
        for name, param in model.named_parameters():
            if name not in peft_param_names and param.requires_grad:
                # Strip LoRA wrapper prefix so keys match the raw model keys
                save_key = name.removeprefix(lora_prefix)
                non_peft_state_dict[save_key] = param.detach().cpu().to(torch.float32)

        # Save as .safetensors to match the reload logic in _load_model_weights
        from safetensors.torch import save_file
        save_file(non_peft_state_dict, os.path.join(checkpoint_folder, "non_peft_params.safetensors"))

        return control

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
        self.dev_text_norm = None
        self.eval_text_norm = None

    def _initialize_container(self):
        """Initialize the model container with appropriate configuration."""
        self.container = DixtralContainer(
            model_args=self.model_args,
            n_last_dec_layers_to_unfreeze=self.training_args.n_last_dec_layers_to_unfreeze,
            use_lora=self.training_args.use_lora,
            params_to_keep_frozen_keywords=self.training_args.params_to_keep_frozen_keywords,
        )

    def _load_training_cutsets(self):
        """Load and prepare training cutsets."""
        train_cutsets = load_cutsets(self.data_args.train_cutsets, self.data_args.use_enrollments)
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
            musan_root=self.aug_args.musan_root,
            musan_augment_prob=self.aug_args.musan_augment_prob,
            text_norm=get_text_norm(self.data_args.train_text_norm),
            feature_extractor=self.container.feature_extractor,
            global_lang_id=self.data_args.global_lang_id,
            load_channel_zero_only=self.data_args.load_channel_zero_only,
            use_enrollments=self.data_args.use_enrollments,
            enrollment_cutset=enrollment_cutset,
        )

        return train_dataset

    def _create_eval_datasets(self, enrollment_cutset):
        """Create development and evaluation datasets."""
        dev_datasets = build_datasets(
            self.data_args.dev_cutsets, self.data_args,
            self.dev_text_norm, self.container, self.data_args.dev_diar_cutsets,
            enrollment_cutset=enrollment_cutset, dataset_class=LhotseLongFormDataset,
            use_ids_as_transcripts=self.data_args.use_diar
        )

        eval_datasets = build_datasets(
            self.data_args.eval_cutsets, self.data_args,
            self.eval_text_norm, self.container, self.data_args.eval_diar_cutsets,
            enrollment_cutset=enrollment_cutset, dataset_class=LhotseLongFormDataset,
            use_ids_as_transcripts=self.data_args.use_diar
        )

        return dev_datasets, eval_datasets

    def _load_model_weights(self):
        """Load pretrained model weights if specified."""
        if self.model_args.reinit_encoder_from:
            enc_state_dict = load_file(self.model_args.reinit_encoder_from)
            enc_state_dict_no_fddt = {k: v for k, v in enc_state_dict.items() if 'fddt' not in k}
            logger.info(self.model.get_encoder().load_state_dict(enc_state_dict_no_fddt, strict=False))

        if self.model_args.reinit_from:
            logger.info(f'Loading model weights from: {self.model_args.reinit_from}')
            path = self.model_args.reinit_from
            if path.endswith('.safetensors'):
                state_dict = load_file(path)
                logger.info(self.model.load_state_dict(state_dict, strict=False))
            else:
                # Load all safetensors files in directory and merge
                state_dict = {}
                for file in os.listdir(path):
                    if file.endswith('.safetensors') and "adapter" not in file:
                        state_dict.update(load_file(os.path.join(path, file)))
                if self.training_args.use_lora:
                    prefix = "base_model.model."
                    state_dict = {prefix + k: v for k, v in state_dict.items()}

                if self.training_args.use_lora:
                    adapter_state_dict = load_file(f"{path}/adapter_model.safetensors")
                    adapter_state_dict = _insert_adapter_name_into_state_dict(adapter_state_dict, "default", "lora_")
                    state_dict = state_dict | adapter_state_dict
                logger.info(self.model.load_state_dict(state_dict, strict=False))
        if self.training_args.soft_prompt_custom_init:
            if hasattr(self.model, "soft_prompt"):
                with torch.no_grad():
                    init_embedding = self.model.language_model.base_model.embed_tokens.weight[34]  # transcribe token
                    self.model.soft_prompt.data.copy_(
                        init_embedding.unsqueeze(0).unsqueeze(0).expand(self.model.soft_prompt.shape).clone())

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
            processor=self.container.processor,
            max_length=self.training_args.generation_max_length,
            model_id=self.model_args.dixtral_base_model,
            prep_for_generate=self.training_args.predict_with_generate,
            num_soft_prompts=self.model_args.num_soft_prompts,
        )

    def _create_compute_metrics_fn(self, dev_datasets):
        """Create metrics computation function."""

        def _compute_metrics(pred, dset=None, split='dev', metrics_list=None):
            step = self.trainer.state.global_step
            output_dir = f'{self.trainer.args.output_dir}/{split}/{step}'
            os.makedirs(output_dir, exist_ok=True)
            return compute_longform_metrics(
                pred, self.trainer, output_dir, self.eval_text_norm,
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

    def _setup_fddt_training(self):
        """Setup FDDT-only training if specified."""
        if (self.training_args.use_fddt_only_n_epochs > 0 or
                self.training_args.use_fddt_only_n_steps > 0):
            self.container.update_model_freezing(self.training_args.prefixes_to_preheat)
        else:
            self.container.update_model_freezing()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

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
        self.collator.prep_for_generate = True
        if decoding_ctc_weight is not None:
            self.model.generation_config.ctc_weight = decoding_ctc_weight

        metrics = self.trainer.evaluate(eval_dataset=eval_datasets, metric_key_prefix=condition_key)
        logger.info(f"Metrics {metrics}")

    def train(self):
        """Main training pipeline."""
        logger.info(f"Config: {self.cfg}")

        # Initialize components
        self._initialize_container()
        self.dev_text_norm = get_text_norm(self.data_args.dev_text_norm)
        self.eval_text_norm = get_text_norm(self.data_args.eval_text_norm)

        # Load data
        train_cutsets = self._load_training_cutsets()
        enrollment_cutset = self._create_enrollment_cutset()
        train_dataset = self._create_train_dataset(train_cutsets, enrollment_cutset)
        dev_datasets, eval_datasets = self._create_eval_datasets(enrollment_cutset)

        # Setup model
        self.model = self.container.model
        self._log_model_parameters()
        self._load_model_weights()
        self._setup_fddt_training()
        update_generation_config(self.model, self.training_args, self.decoding_args,
                                 predict_timestamps=self.data_args.use_timestamps)

        # Create trainer
        self.collator = self._create_data_collator()
        callbacks = [EarlyStoppingCallback(
            self.training_args.early_stopping_patience)] if self.training_args.early_stopping_patience > 0 else []
        if self.training_args.use_lora:
            callbacks.append(SaveNonPeftParamsCallback())
        self.model.is_parallelizable = True
        self.model.model_parallel = True
        self.trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            eval_dataset=dev_datasets,
            data_collator=self.collator,
            train_dataset=train_dataset,
            processing_class=self.container.tokenizer,
            container=self.container,
            callbacks=callbacks,
        )

        # Setup additional components
        self._setup_wandb()

        for p in self.model.parameters():
            p.data = p.data.to(torch.bfloat16)

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
