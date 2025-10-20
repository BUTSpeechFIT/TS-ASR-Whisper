from typing import Any, Union, Dict, List, Optional, Tuple

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import logging

from utils.compute_overall_statisctics import main as compute_overall_stats

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


class GradLogger(TrainerCallback):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if wandb.run is not None:
            wandb.watch(self.model, log='all', log_freq=50)
        else:
            raise ValueError("wandb is not initialized")


class CustomTrainerEncoder(Trainer):
    def __init__(self, container, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_w_cast = None
        self.forward_wo_cast = None
        self.container = container
        self.chunk_length = None
        if hasattr(self.model, "get_max_len"):
            self.chunk_length = self.model.get_max_len()

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        labels = inputs.pop("labels")
        # If input sequence is longer then model maximum length
        length = inputs[model.main_input_name].size(-1)

        if self.chunk_length is not None and length > self.chunk_length:
            feats = inputs[model.main_input_name].split(self.chunk_length, dim=-1)
            att_masks = inputs['attention_mask'].split(self.chunk_length, dim=-1)
            inputs_ = zip(feats, att_masks)
            logits = []
            for inputs_local, mask in inputs_:
                logits.append(super().prediction_step(
                    model, {model.main_input_name: inputs_local, "attention_mask": mask, 'return_logits': True}, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
                )[1][0])
            logits = torch.concat(logits, dim=1)

        else:
            inputs['return_logits'] = True
            output = super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
            )
            logits = output[1][0]
        loss = self.model.get_loss(logits, labels)

        output = (loss, logits, labels)
        return output

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")

        outputs = model(**inputs, return_logits=True)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        for token in self.processing_class.prefix_tokens:
            if (labels[:, 0] == token).all():
                labels = labels[:, 1:]
        labels[labels == self.processing_class.eos_token_id] = -100

        loss = self.model.get_loss(outputs.logits, labels)

        return (loss, outputs) if return_outputs else loss


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, container, model, params_to_keep_frozen, **kwargs):
        super().__init__(model=model, **kwargs)
        self.forward_w_cast = None
        self.forward_wo_cast = None
        self.container = container
        self.warmup_phase = True
        self.params_to_keep_frozen = params_to_keep_frozen
        self.metric_key_prefix = ""

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.warmup_phase and self.state.epoch >= self.args.use_fddt_only_n_epochs and self.state.global_step >= self.args.use_fddt_only_n_steps:
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                    continue
                for keyword in self.params_to_keep_frozen:
                    if keyword in name:
                        param.requires_grad = False
                        break
                else:
                    param.requires_grad = True
            logger.info(f"***** Unfreezing params except {self.params_to_keep_frozen}*****")
            logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")
            self.create_optimizer_and_scheduler(num_training_steps=self.state.max_steps)

            self.warmup_phase = False
        output = super().training_step(model, inputs, num_items_in_batch)
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

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        self.metric_key_prefix = metric_key_prefix
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        return output


    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # We want to disable loss computation as it is not ready for longform input
        labels = inputs.pop("labels")
        gen_config = self.model.generation_config

        if self.args.bf16_full_eval:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, generated_tokens, _ = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)
        else:
            loss, generated_tokens, _ = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys,
                                                                **gen_kwargs)

        if labels is not None:
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        return loss, generated_tokens, labels

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs,
    ) -> Dict[str, float]:
        output = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        if self.args.compute_combined_metrics and (eval_dataset is None or isinstance(eval_dataset, dict)):
            step = self.state.global_step
            output_dir = f'{self.args.output_dir}/*/{step}'
            overall_stats = compute_overall_stats(output_dir,
                                                  f"{self.args.output_dir}/{metric_key_prefix}_{step}_comined_tcp_wer.csv")
            overall_stats_dict = overall_stats.squeeze().to_dict()
            overall_stats_dict = {f"{metric_key_prefix}_overall_{key}": val for key, val in overall_stats_dict.items()
                                  if key != "source_language"}
            self.log(overall_stats_dict)
            output |= overall_stats_dict
        return output
