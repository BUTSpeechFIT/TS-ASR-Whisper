import re

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers.models.whisper import WhisperFeatureExtractor, WhisperTokenizerFast
from transformers.utils import logging

from models.dicow.modeling_dicow import DiCoWForConditionalGeneration

logger = logging.get_logger("transformers")

PEFT_WRAPPER_PREFIX = "base_model.model."
# modules_to_save wraps a module and inserts .modules_to_save.<adapter>. / .original_module.
# into every param path under it, e.g. model.encoder.fddts... -> model.encoder.modules_to_save.default.fddts...
_MODULES_TO_SAVE_INFIX_RE = re.compile(r"\.(?:modules_to_save\.[^.]+|original_module)\.")


def unwrap_param_name(model, name):
    """Strip PEFT-wrapping segments so prefix lists like prefixes_to_preheat keep matching."""
    if not isinstance(model, PeftModel):
        return name
    if name.startswith(PEFT_WRAPPER_PREFIX):
        name = name[len(PEFT_WRAPPER_PREFIX):]
    return _MODULES_TO_SAVE_INFIX_RE.sub(".", name)


class WhisperContainer:
    def __init__(self, params_to_keep_frozen_keywords=None, remove_timestamps_from_ctc=False,
                 model_args=None, data_args=None, use_fddt=False, use_lora=False):
        self.model_type = model_args.whisper_model
        predict_timestamps = data_args.use_timestamps
        global_lang_id = data_args.global_lang_id
        overwrite_args = {
            "ctc_weight": model_args.ctc_weight,
            "fddt_is_diagonal": model_args.fddt_is_diagonal,
            "fddt_bias_only": model_args.fddt_bias_only,
            "fddt_use_silence": model_args.fddt_use_silence,
            "fddt_use_target": model_args.fddt_use_target,
            "fddt_use_overlap": model_args.fddt_use_overlap,
            "fddt_use_non_target": model_args.fddt_use_non_target,
            "remove_timestamps_from_ctc": remove_timestamps_from_ctc,
            "apply_fddt_to_n_layers": model_args.apply_fddt_to_n_layers,
            "use_fddt": use_fddt,
            "fddt_init": model_args.fddt_init,
            "non_target_fddt_value": model_args.non_target_fddt_value,
            "use_pre_pos_fddt": model_args.use_pre_pos_fddt,
            "use_enrollments": data_args.use_enrollments,
            "pre_ctc_sub_sample": model_args.pre_ctc_sub_sample,
            "ctc_length_margin": model_args.ctc_length_margin,
            "additional_layer": model_args.additional_layer,
            "additional_self_attention_layer": model_args.additional_self_attention_layer,
            "scb_layers": model_args.scb_layers,
        }
        clean_kwargs = {k: v for k, v in overwrite_args.items() if v is not None}
        self.model = DiCoWForConditionalGeneration.from_pretrained(
            self.model_type,
            **clean_kwargs
        )

        self.model.post_init()
        self._init_ctc_head_from_decoder_embeddings()

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_type)
        self.tokenizer = WhisperTokenizerFast.from_pretrained(self.model_type, predict_timestamps=predict_timestamps)

        if ".en" not in self.model_type:
            self.model.generation_config.language = None
            self.model.generation_config.task = "transcribe"
            # This ensures labels
            self.tokenizer.set_prefix_tokens(predict_timestamps=predict_timestamps, task="transcribe",
                                             language=global_lang_id)
        else:
            self.tokenizer.set_prefix_tokens(predict_timestamps=predict_timestamps)

        self.model.set_tokenizer(self.tokenizer)
        self.model.config.forced_decoder_ids = None

        if use_lora:
            lora_config = LoraConfig(
                r=16,  # LoRA rank (tune as needed)
                lora_alpha=32,  # LoRA alpha (scaling)
                target_modules=r".*decoder.*(q_proj|k_proj|v_proj|out_proj|fc1|fc2).*",
                lora_dropout=0.0,
                bias="none",
                modules_to_save=['encoder']
            )

            self.model = get_peft_model(self.model, lora_config)

        if params_to_keep_frozen_keywords is not None:
            self.freeze_by_keywords(params_to_keep_frozen_keywords)

    def _init_ctc_head_from_decoder_embeddings(self):
        """Initialize the CTC head (encoder.lm_head) from the decoder's output token
        embeddings instead of leaving it randomly initialized: both operate over the
        same vocabulary, so this gives CTC (pre)training a much better starting point
        than random noise. The CTC head has one extra output -- the blank token,
        appended as the last index (see encoder.py's `blank=logits.shape[-1] - 1`) --
        which has no decoder counterpart and is left at its random init."""
        encoder = self.model.get_encoder()
        if not hasattr(encoder, "lm_head"):
            return
        decoder_embed = self.model.get_output_embeddings().weight
        vocab_size = decoder_embed.shape[0]
        with torch.no_grad():
            encoder.lm_head.weight[:vocab_size].copy_(decoder_embed)

    def freeze_except(self, prefixes_to_preheat):
        for name, param in self.model.named_parameters():
            if "original_module" in name:
                param.requires_grad = False
                continue
            unwrapped = unwrap_param_name(self.model, name)
            param.requires_grad = False
            for prefix in prefixes_to_preheat:
                if unwrapped.startswith(prefix):
                    param.requires_grad = True

    def freeze_by_keywords(self, params_to_keep_frozen_keywords):
        """Make everything trainable except params matching one of the keywords.

        This is the default trainable/frozen layout of the model; it is applied at init and
        re-applied after the FDDT preheat phase to undo `freeze_except`.
        """
        for name, param in self.model.named_parameters():
            if "original_module" in name:
                param.requires_grad = False
                continue
            if "lora_" in name:
                param.requires_grad = True
                continue
            for keyword in params_to_keep_frozen_keywords:
                if keyword in name:
                    param.requires_grad = False
                    break
            else:
                param.requires_grad = True


class NaNSafeAdamW(torch.optim.AdamW):
    """AdamW that skips the update instead of applying it when gradients are non-finite.

    bf16 training has no GradScaler-style safety net (that only exists for fp16), so a single
    batch producing an inf/nan gradient would otherwise be applied unconditionally and
    permanently corrupt the model -- every step after that stays nan for the rest of the job.
    After too many consecutive skips, training is aborted instead of silently burning the rest
    of the allocation on a dead model.
    """
    max_consecutive_skips = 20

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._consecutive_skips = 0

    def step(self, closure=None):
        finite = all(
            torch.isfinite(p.grad).all()
            for group in self.param_groups
            for p in group['params']
            if p.grad is not None
        )
        if not finite:
            self._consecutive_skips += 1
            logger.warning(
                f"Non-finite gradient detected, skipping optimizer step "
                f"({self._consecutive_skips}/{self.max_consecutive_skips})."
            )
            self.zero_grad(set_to_none=True)
            if self._consecutive_skips >= self.max_consecutive_skips:
                raise RuntimeError(
                    f"Non-finite gradients for {self.max_consecutive_skips} consecutive steps; aborting training."
                )
            return None
        self._consecutive_skips = 0
        return super().step(closure)


def get_optimizer(model, training_args, prefixes_with_higher_lr=None):
    if prefixes_with_higher_lr is None:
        prefixes_with_higher_lr = []
    if training_args.use_custom_optimizer:
        original_whisper_params = [param for name, param in model.named_parameters() if
                                   param.requires_grad and not any([unwrap_param_name(model, name).startswith(prefix) for prefix in prefixes_with_higher_lr])]
        new_params = [param for name, param in model.named_parameters() if
                      param.requires_grad and any([unwrap_param_name(model, name).startswith(prefix) for prefix in prefixes_with_higher_lr])]
        return NaNSafeAdamW([{'params': original_whisper_params},
                             {'params': new_params,
                              'lr': training_args.fddt_lr_multiplier * training_args.learning_rate,
                              'weight_decay': 0.0}],
                            lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    else:
        return None
