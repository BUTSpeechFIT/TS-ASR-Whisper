import torch
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
    def __init__(self, use_flash_attention=False, params_to_keep_frozen_keywords=None, remove_timestamps_from_ctc=False, model_args=None, data_args=None, use_fddt=False, use_bf16=False):
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
                                       use_initial_fddt=model_args.use_initial_fddt,
                                       uses_enrollments=data_args.use_enrollments,
                                       pre_ctc_sub_sample=model_args.pre_ctc_sub_sample,
                                       additional_layer=model_args.additional_layer,
                                       additional_self_attention_layer=model_args.additional_self_attention_layer,
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
        self.model.generation_config.forced_decoder_ids = None
        self.model.config.forced_decoder_ids = None

        if predict_timestamps:
            self.model.generation_config.return_timestamps = predict_timestamps

        if params_to_keep_frozen_keywords is not None:
            for name, param in self.model.named_parameters():
                for keyword in params_to_keep_frozen_keywords:
                    if keyword in name:
                        param.requires_grad = False

    def freeze_except(self, prefixes_to_preheat):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for prefix in prefixes_to_preheat:
                if name.startswith(prefix):
                    param.requires_grad = True


def get_optimizer(model, training_args, prefixes_with_higher_lr=None):
    if prefixes_with_higher_lr is None:
        prefixes_with_higher_lr = []
    if training_args.use_custom_optimizer:
        original_whisper_params = [param for name, param in model.named_parameters() if
                                   not any([name.startswith(prefix) for prefix in prefixes_with_higher_lr])]
        new_params = [param for name, param in model.named_parameters() if
                      any([name.startswith(prefix) for prefix in prefixes_with_higher_lr])]
        return torch.optim.AdamW([{'params': original_whisper_params},
                                  {'params': new_params,
                                   'lr': training_args.fddt_lr_multiplier * training_args.learning_rate,
                                   'weight_decay': 0.0}],
                                 lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    else:
        return None
