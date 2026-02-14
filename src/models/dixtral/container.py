import re

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor
from transformers.utils import logging

from src.models.dicow.modeling_dicow import DiCoWForConditionalGeneration, DiCoWConfig
from src.models.dixtral.modeling_dixtral import DixtralForConditionalGeneration, DixtralConfig

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")

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
        logger.info("✓ Copied initial_fddt")

    # Copy FDDT layers
    if hasattr(dixtral_encoder, 'fddts') and hasattr(dicow_encoder, 'fddts'):
        num_fddts = min(len(dixtral_encoder.fddts), len(dicow_encoder.fddts))
        for i in range(num_fddts):
            dixtral_encoder.fddts[i].load_state_dict(
                dicow_encoder.fddts[i].state_dict()
            )
        logger.info(f"✓ Copied {num_fddts} FDDT layers")

    logger.info("\n✅ All DiCoW weights copied successfully!")

class DixtralContainer:
    def __init__(self, params_to_keep_frozen_keywords=None, n_last_dec_layers_to_unfreeze=0,
                 model_args=None, use_lora=False):
        model_id = model_args.dixtral_base_model

        config = DixtralConfig.from_pretrained(
            model_id,
            num_soft_prompts=model_args.num_soft_prompts,
        )

        if model_args.dixtral_load_fddt_from or model_args.dixtral_replace_encoder_from:
            dicow_audio_config = DiCoWConfig.from_pretrained(model_args.dixtral_load_fddt_from or model_args.dixtral_replace_encoder_from)
            for key, value in dicow_audio_config.to_dict().items():
                if hasattr(config.audio_config, key):
                    setattr(config.audio_config, key, value)

            # Then override specific values
            config.audio_config.use_dicow_encoder = True
            # config.audio_config.ctc_weight = 0.0
            # config.audio_config.additional_layer = False
            # config.audio_config.additional_self_attention_layer = False
            # config.audio_config.pre_ctc_sub_sample = False


        self.model = DixtralForConditionalGeneration.from_pretrained(
            model_id,
            config=config,
            ignore_mismatched_sizes=True,  # For new DiCoW components
        )


        if model_args.dixtral_load_fddt_from:
            # Copy the weights
            dicow = DiCoWForConditionalGeneration.from_pretrained(model_args.dixtral_load_fddt_from)
            copy_fddt_weights(self.model, dicow)
            del dicow
        if model_args.dixtral_replace_encoder_from:
            dicow = DiCoWForConditionalGeneration.from_pretrained(model_args.dixtral_replace_encoder_from)
            dixtral_encoder = self.model.audio_tower
            dicow_encoder = dicow.model.encoder
            logger.info(dixtral_encoder.load_state_dict(dicow_encoder.state_dict(), strict=False))

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer


        self.model.set_tokenizer(self.processor.tokenizer)
        self.model.config.forced_decoder_ids = None

        if use_lora:
            lora_config = LoraConfig(
                r=64,
                lora_alpha=32,
                target_modules=r".*language_model.*(q_proj|k_proj|v_proj|o_proj|down_proj|up_proj|gate_proj).*",
                lora_dropout=0.0,
                bias="none",
            )

            self.model = get_peft_model(self.model, lora_config)

        # 1. SAVE THE KEYWORDS to self
        self.params_to_keep_frozen = params_to_keep_frozen_keywords

        self.n_last_dec_layers_to_unfreeze = n_last_dec_layers_to_unfreeze
        self.prefixes_to_keep_training = []  # Stores the preheat prefixes persistently

    def update_model_freezing(self, prefixes_to_preheat=None):
        """
        Master freezing controller.

        Args:
            prefixes_to_preheat (list): If provided, enters PHASE 1 (Preheat).
                                        These prefixes are saved and will KEEP training in Phase 2.
                                        If None, enters PHASE 2 (Finetune).
        """

        # --- 1. STATE MANAGEMENT ---
        # If prefixes are provided, we are entering/in Phase 1. Save them.
        if prefixes_to_preheat is not None:
            self.prefixes_to_keep_training = prefixes_to_preheat
            is_phase_1_strict = True
        else:
            # If None passed, we are in Phase 2, but we keep using the saved prefixes
            is_phase_1_strict = False

        # --- 2. SETUP CALCULATIONS ---
        n_layers = self.n_last_dec_layers_to_unfreeze or 0

        # Calculate LLM Cutoff
        cutoff_layer = 9999
        if hasattr(self.model, "language_model"):
            llm_config = self.model.language_model.config
            total_layers = getattr(llm_config, "num_hidden_layers", getattr(llm_config, "n_layer", 32))
            cutoff_layer = total_layers - n_layers

        logger.info(f"Freezing Update | Phase: {'1 (Preheat)' if is_phase_1_strict else '2 (Finetune)'}")

        # --- 3. PARAMETER LOOP ---
        trainable_count = 0

        for name, param in self.model.named_parameters():

            # A. GLOBAL BLACKLIST (Highest Priority)
            # If it's in the blacklist, it NEVER trains.
            # This fixes the 'audio_tower.conv1' issue.
            is_blacklisted = (self.params_to_keep_frozen is not None and
                              any(k in name for k in self.params_to_keep_frozen))

            # B. LORA (Always Trains)
            if "lora_" in name:
                param.requires_grad = True
                trainable_count += param.numel()
                continue

            # C. PREHEAT WHITELIST (Persistent)
            # If this module was preheated, it MUST stay training in Phase 2.
            is_preheat_module = any(name.startswith(p) for p in self.prefixes_to_keep_training)

            if is_preheat_module:
                param.requires_grad = True
                trainable_count += param.numel()
                continue

            # --- D. PHASE-SPECIFIC LOGIC ---

            if is_phase_1_strict:
                # PHASE 1: If we reached here, it wasn't LoRA and wasn't in the whitelist.
                # So it must be FROZEN.
                param.requires_grad = False

            else:
                # PHASE 2: Standard Finetuning Logic
                if "language_model" in name:
                    # Check Layer Index
                    layer_match = re.search(r"layers\.(\d+)\.", name)
                    if layer_match and (int(layer_match.group(1)) >= cutoff_layer):
                        should_train = True
                    else:
                        should_train = not is_blacklisted

                    param.requires_grad = should_train
                else:
                    # Default for other components (that weren't preheated or blacklisted)
                    # e.g., unused adapters or new heads. Default to TRAIN.
                    param.requires_grad = not is_blacklisted

            if param.requires_grad:
                trainable_count += param.numel()

        logger.info(f"Freezing update complete. Total Trainable Params: {trainable_count:,}")