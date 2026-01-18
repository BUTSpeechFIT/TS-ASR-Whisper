from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor
import torch
from src.models.dicow.modeling_dicow import DiCoWForConditionalGeneration, DiCoWConfig
from src.models.dixtral.modeling_dixtral import DixtralForConditionalGeneration, DixtralConfig


# Copy FDDT parameters
def copy_fddt_weights(dixtral_model, dicow_model):
    """Copy FDDT and related DiCoW weights from DiCoW encoder to Dixtral encoder."""
    dixtral_encoder = dixtral_model.audio_tower
    dicow_encoder = dicow_model.model.encoder

    # Copy initial FDDT if exists
    if hasattr(dixtral_encoder, 'initial_fddt') and hasattr(dicow_encoder, 'initial_fddt'):
        dixtral_encoder.initial_fddt.load_state_dict(
            dicow_encoder.initial_fddt.state_dict()
        )
        print("✓ Copied initial_fddt")

    # Copy FDDT layers
    if hasattr(dixtral_encoder, 'fddts') and hasattr(dicow_encoder, 'fddts'):
        num_fddts = min(len(dixtral_encoder.fddts), len(dicow_encoder.fddts))
        for i in range(num_fddts):
            dixtral_encoder.fddts[i].load_state_dict(
                dicow_encoder.fddts[i].state_dict()
            )
        print(f"✓ Copied {num_fddts} FDDT layers")

    print("\n✅ All DiCoW weights copied successfully!")

class DixtralContainer:
    def __init__(self, params_to_keep_frozen_keywords=None, remove_timestamps_from_ctc=False,
                 model_args=None, use_lora=False):
        model_id = model_args.dixtral_base_model

        config = DixtralConfig.from_pretrained(
            model_id,
        )

        if model_args.dixtral_load_fddt_from:
            dicow_audio_config = DiCoWConfig.from_pretrained(model_args.dixtral_load_fddt_from)
            for key, value in dicow_audio_config.to_dict().items():
                if hasattr(config.audio_config, key):
                    setattr(config.audio_config, key, value)

            # Then override specific values
            config.audio_config.use_dicow_encoder = True
            config.audio_config.ctc_weight = 0.1
            config.audio_config.additional_layer = True
            config.audio_config.additional_self_attention_layer = False
            config.audio_config.pre_ctc_sub_sample = False


        self.model = DixtralForConditionalGeneration.from_pretrained(
            model_id,
            config=config,
            ignore_mismatched_sizes=True  # For new DiCoW components
        )


        if model_args.dixtral_load_fddt_from:
            # Copy the weights
            dicow = DiCoWForConditionalGeneration.from_pretrained(model_args.dixtral_load_fddt_from)
            copy_fddt_weights(self.model, dicow)
            del dicow

        # Copy language model head weights to CTC head if CTC is enabled
        if (config.audio_config.use_dicow_encoder and
                config.audio_config.ctc_weight > 0.0):
            embed_tokens = self.model.language_model.model.embed_tokens
            ctc_lm_head = self.model.ctc_lm_head  # Fixed: it's in the main model, not audio_tower

            with torch.no_grad():
                # Copy all weights except the blank token (last row)
                ctc_lm_head.weight.data[:-1] = embed_tokens.weight.data.to(
                    device=ctc_lm_head.weight.device,
                    dtype=ctc_lm_head.weight.dtype
                )

            print("Copied LM embed_tokens weights to CTC LM head")

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer


        self.model.set_tokenizer(self.processor.tokenizer)
        self.model.config.forced_decoder_ids = None

        if use_lora:
            lora_config = LoraConfig(
                r=64,
                lora_alpha=32,
                target_modules=r".*language_model.*(q_proj|k_proj|v_proj|o_proj|down_proj|up_proj).*",
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
