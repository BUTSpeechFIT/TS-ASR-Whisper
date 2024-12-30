from types import MethodType
from typing import Any, Union, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import Seq2SeqTrainer, Trainer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import logging
from transformers.trainer_pt_utils import get_model_param_count

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


class CustomTrainerEncoder(Trainer):
    def __init__(self, container, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_w_cast = None
        self.forward_wo_cast = None
        self.container = container

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if hasattr(self.container, 'h'):
            self.container.h.set_diar_output(inputs['vad_mask'])

        labels = inputs.pop("labels")

        # else:
        output = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        loss = self.model.get_loss(output[1], labels)

        output = (loss, output[1], labels)
        return output

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        for token in self.tokenizer.prefix_tokens:
            if (labels[:, 0] == token).all():
                labels = labels[:, 1:]
        labels[labels == self.tokenizer.eos_token_id] = -100

        loss = self.model.get_loss(outputs.logits, labels)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if hasattr(self.container, 'h'):
            self.container.h.set_diar_output(inputs['vad_mask'])
        out = super().training_step(model, inputs)
        return out


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, container, *args, params_to_keep_frozen, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_w_cast = None
        self.forward_wo_cast = None
        self.container = container
        self.warmup_phase = True
        self.params_to_keep_frozen = params_to_keep_frozen

    def prediction_step_local(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = False
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
                "labels" in generation_inputs
                and "decoder_input_ids" in generation_inputs
                and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }
        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        labels = inputs["labels"]
        if labels.shape[-1] < gen_config.max_length:
            labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
            labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)

        return loss, generated_tokens, labels

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_valid_field = "is_valid" in inputs
        if has_valid_field:
            for key in inputs.keys():
                if key == "is_valid" or key == "per_group_sizes":
                    continue
                inputs[key] = inputs[key][inputs["is_valid"]]
            is_valid = inputs.pop("is_valid")
        if hasattr(self.container, 'h'):
            self.container.h.set_diar_output(inputs['vad_mask'])

        if self.args.bf16_full_eval and not prediction_loss_only:
            forward = model.forward
            original_forward = model.__dict__.pop("_original_forward", None)
            if original_forward is not None:
                self.forward_w_cast = forward
                while hasattr(forward, "__wrapped__"):
                    forward = forward.__wrapped__
                    if forward == original_forward:
                        break
                self.forward_wo_cast = MethodType(forward, model)

            model.forward = self.forward_wo_cast

            with torch.autocast(dtype=torch.bfloat16, device_type=self.model.device.type):
                output = self.prediction_step_local(
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
                )
            model.forward = self.forward_w_cast
        else:
            output = self.prediction_step_local(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
            )
        if has_valid_field:
            loss, generated_tokens, labels = output
            generated_tokens_original = torch.full((is_valid.shape[0], generated_tokens.shape[1]), -100,
                                                   dtype=torch.long, device=generated_tokens.device)
            generated_tokens_original[is_valid] = generated_tokens
            labels_original = torch.full((is_valid.shape[0], labels.shape[1]), -100, dtype=torch.long,
                                         device=labels.device)
            labels_original[is_valid] = labels
            output = (loss, generated_tokens_original, labels_original)
        return output

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if "is_valid" in inputs:
            for key in inputs.keys():
                if key == "is_valid" or key == "per_group_sizes":
                    continue
                inputs[key] = inputs[key][inputs["is_valid"]]
            inputs.pop("is_valid")

        if hasattr(self.container, 'h'):
            self.container.h.set_diar_output(inputs['vad_mask'])
        output = super().training_step(model, inputs)
        if self.warmup_phase and self.state.epoch >= self.args.use_amplifiers_only_n_epochs and self.state.global_step >= self.args.use_amplifiers_only_n_steps:
            for name, param in self.model.named_parameters():
                param.requires_grad = True
                for keyword in self.params_to_keep_frozen:
                    if keyword in name:
                        param.requires_grad = False
            logger.info(f"***** Unfreezing params except {self.params_to_keep_frozen}*****")
            logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

            self.warmup_phase = False
        return output

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
                if self.state.train_batch_size is not None:
                    self.args.gradient_accumulation_steps *= (self.state.train_batch_size // self._train_batch_size)
                    if args is not None:
                        args.gradient_accumulation_steps = self.args.gradient_accumulation_steps
            self.state.train_batch_size = self._train_batch_size
        out = super()._inner_training_loop(
            batch_size=batch_size, args=args, resume_from_checkpoint=resume_from_checkpoint, trial=trial
        )
        return out
