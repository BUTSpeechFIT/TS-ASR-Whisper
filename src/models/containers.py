import torch
from peft import LoraConfig, get_peft_model
from transformers.models.whisper import WhisperFeatureExtractor, WhisperTokenizerFast

from models.dicow.modeling_dicow import DiCoWForConditionalGeneration


def supports_flash_attention():
    """Check if a GPU supports FlashAttention."""
    major, minor = torch.cuda.get_device_capability()

    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

    return is_sm8x or is_sm90


class WhisperContainer:
    def __init__(self, use_flash_attention=False, params_to_keep_frozen_keywords=None, remove_timestamps_from_ctc=False,
                 model_args=None, data_args=None, use_fddt=False, use_lora=False):
        self.model_type = model_args.whisper_model
        predict_timestamps = data_args.use_timestamps
        global_lang_id = data_args.global_lang_id
        self.model = (DiCoWForConditionalGeneration
                      .from_pretrained(self.model_type,
                                       attn_implementation="flash_attention_2" if torch.cuda.is_available() and supports_flash_attention() and use_flash_attention else None,
                                       ctc_weight=model_args.ctc_weight,
                                       fddt_is_diagonal=model_args.fddt_is_diagonal,
                                       fddt_bias_only=model_args.fddt_bias_only,
                                       fddt_use_silence=model_args.fddt_use_silence,
                                       fddt_use_target=model_args.fddt_use_target,
                                       fddt_use_overlap=model_args.fddt_use_overlap,
                                       fddt_use_non_target=model_args.fddt_use_non_target,
                                       remove_timestamps_from_ctc=remove_timestamps_from_ctc,
                                       apply_fddt_to_n_layers=model_args.apply_fddt_to_n_layers,
                                       use_fddt=use_fddt,
                                       fddt_init=model_args.fddt_init,
                                       non_target_fddt_value=model_args.non_target_fddt_value,
                                       use_pre_pos_fddt=model_args.use_pre_pos_fddt,
                                       use_enrollments=data_args.use_enrollments,
                                       pre_ctc_sub_sample=model_args.pre_ctc_sub_sample,
                                       additional_layer=model_args.additional_layer,
                                       additional_self_attention_layer=model_args.additional_self_attention_layer,
                                       scb_layers=model_args.scb_layers,
                                       )

                      )
        self.model.post_init()

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
            )

            self.model = get_peft_model(self.model, lora_config)

        if params_to_keep_frozen_keywords is not None:
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                    continue
                for keyword in params_to_keep_frozen_keywords:
                    if keyword in name:
                        param.requires_grad = False
                        break
                else:
                    param.requires_grad = True

    def freeze_except(self, prefixes_to_preheat):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for prefix in prefixes_to_preheat:
                if name.startswith(prefix):
                    param.requires_grad = True

def get_optimizer(model, training_args, prefixes_with_higher_lr=None):
    """
    Returns an AdamW optimizer with support for differential learning rates
    (higher LR for specific parameter prefixes).
    """
    if prefixes_with_higher_lr is None:
        prefixes_with_higher_lr = []

    # If the user has a flag for custom optimizer logic, use it.
    # Otherwise, you might return None to let Trainer use its default.
    if getattr(training_args, "use_custom_optimizer", True):

        adam_params_standard = []
        adam_params_new = []

        for name, param in model.named_parameters():
            # [MEMORY SAVER] Skip frozen parameters
            if not param.requires_grad:
                continue

            # Identify if this is a "high LR" parameter based on prefix
            is_new_param = any(name.startswith(prefix) for prefix in prefixes_with_higher_lr)

            # Sort parameters into groups
            if is_new_param:
                adam_params_new.append(param)
            else:
                adam_params_standard.append(param)

        # --- Create Parameter Groups ---
        optimizer_grouped_parameters = []

        # Group 1: Standard Parameters (Base LR)
        if adam_params_standard:
            optimizer_grouped_parameters.append({
                'params': adam_params_standard,
                'lr': training_args.learning_rate,
                'weight_decay': training_args.weight_decay
            })

        # Group 2: New/Specific Parameters (Higher LR)
        if adam_params_new:
            # Ensure multiplier exists in args, default to 1.0 if missing
            multiplier = getattr(training_args, "fddt_lr_multiplier", 1.0)
            optimizer_grouped_parameters.append({
                'params': adam_params_new,
                'lr': multiplier * training_args.learning_rate,
                'weight_decay': training_args.weight_decay
            })

        # --- Initialize Standard AdamW ---
        if len(optimizer_grouped_parameters) > 0:
            return torch.optim.AdamW(optimizer_grouped_parameters)
        else:
            return None

    return None