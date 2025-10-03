from dataclasses import dataclass
from typing import Optional

import torch
from transformers import WhisperConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, Seq2SeqModelOutput


@dataclass
class Seq2SeqLMOutputLosses(Seq2SeqLMOutput):
    enc_loss: Optional[torch.FloatTensor] = None
    dec_loss: Optional[torch.FloatTensor] = None
    encoder_logits: Optional[torch.FloatTensor] = None


@dataclass
class BaseModelOutputLogit(BaseModelOutput):
    logits: Optional[torch.FloatTensor] = None


@dataclass
class Seq2SeqModelOutputLogit(Seq2SeqModelOutput):
    encoder_logits: Optional[torch.FloatTensor] = None


class DiCoWConfig(WhisperConfig):
    """This is a modified version of the `WhisperEncoder` model from the `transformers` library.
    The model has been modified to support CTC loss computation in the forward pass."""
    model_type = "DiCoW"

    def __init__(
            self,
            ctc_loss_reduction: str = "mean",
            final_dropout: float = 0.0,
            ctc_zero_infinity: bool = False,
            ctc_weight: float = 0.0,
            blank_token_id: Optional[int] = None,
            additional_layer: bool = False,
            additional_self_attention_layer: bool = False,
            sub_sample: bool = False,
            use_fddt: bool = True,
            fddt_is_diagonal: bool = True,
            fddt_bias_only: bool = False,
            fddt_use_silence: bool = True,
            fddt_use_target: bool = True,
            fddt_use_overlap: bool = True,
            fddt_use_non_target: bool = True,
            remove_timestamps_from_ctc: bool = False,
            apply_fddt_to_n_layers: int = -1,
            fddt_init: str = 'suppressive',  # random, non-disturbing
            mt_num_speakers: int = 1,
            is_mt: bool = False,
            non_target_fddt_value: float = 0.0,
            use_initial_fddt: bool = False,
            scb_method: str = None,
            scb_layers: int = -1,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.ctc_loss_reduction = ctc_loss_reduction
        self.final_dropout = final_dropout
        self.ctc_zero_infinity = ctc_zero_infinity
        self.ctc_weight = ctc_weight
        self.blank_token_id = blank_token_id
        self.additional_layer = additional_layer
        self.additional_self_attention_layer = additional_self_attention_layer
        self.sub_sample = sub_sample
        self.use_fddt = use_fddt
        self.fddt_is_diagonal = fddt_is_diagonal
        self.fddt_bias_only = fddt_bias_only
        self.fddt_use_silence = fddt_use_silence
        self.fddt_use_target = fddt_use_target
        self.fddt_use_overlap = fddt_use_overlap
        self.fddt_use_non_target = fddt_use_non_target
        self.remove_timestamps_from_ctc = remove_timestamps_from_ctc
        self.apply_fddt_to_n_layers = apply_fddt_to_n_layers
        self.fddt_init = fddt_init
        self.mt_num_speakers = mt_num_speakers
        self.non_target_fddt_value = non_target_fddt_value
        self.use_initial_fddt = use_initial_fddt
        self.scb_method = scb_method
        self.scb_layers = scb_layers
        self.is_mt = is_mt



_HIDDEN_STATES_START_POSITION = 2
