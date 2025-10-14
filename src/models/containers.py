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
    def __init__(self, model_type='whisper-tiny', ctc_weight=0.0,
                 training_args=None, predict_timestamps=False, global_lang_id="en", params_to_keep_frozen_keywords=None,
                 **kwargs):
        self.model_type = model_type
        self.model = (DiCoWForConditionalGeneration
                      .from_pretrained(model_type,
                                       device_map='cpu',
                                       low_cpu_mem_usage=True,
                                       use_safetensors=True,
                                       attn_implementation="flash_attention_2" if torch.cuda.is_available() and supports_flash_attention() and training_args.bf16 else None,
                                       sub_sample=True,
                                       additional_self_attention_layer=True,
                                       ctc_weight=ctc_weight,
                                       **kwargs
                                       )

                      )
        self.model.post_init()

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_type)
        self.tokenizer = WhisperTokenizerFast.from_pretrained(self.model_type, predict_timestamps=predict_timestamps)

        if ".en" not in model_type:
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

        self.model.generation_config.ctc_weight = ctc_weight
        self.model.generation_config.ctc_margin = 0

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
