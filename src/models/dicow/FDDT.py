from typing import Optional

import torch
from torch import nn

from .layers import CustomDiagonalLinear, CustomLinear
from .SCBs import SpeakerCommunicationBlock


class FDDT(nn.Module):
    def __init__(self, config, d_model, non_target_rate=0.01, is_diagonal=False, bias_only=False, use_silence=True,
                 use_target=True, use_overlap=True, use_non_target=True, use_interaction=False):
        super().__init__()
        if use_target:
            self.target_linear = nn.Parameter(torch.zeros(d_model)) if bias_only else (
                CustomDiagonalLinear(d_model, bias=True, init_eye_val=1.0) if is_diagonal else CustomLinear(d_model,
                                                                                                            d_model,
                                                                                                            bias=True,
                                                                                                            init_eye_val=1.0))
        if use_non_target:
            self.non_target_linear = nn.Parameter(torch.zeros(d_model)) if bias_only else (
                CustomDiagonalLinear(d_model, bias=True, init_eye_val=non_target_rate) if is_diagonal else CustomLinear(
                    d_model, d_model, bias=True, init_eye_val=non_target_rate))
        if use_overlap:
            self.overlap_linear = nn.Parameter(torch.zeros(d_model)) if bias_only else (
                CustomDiagonalLinear(d_model, bias=True, init_eye_val=1.0) if is_diagonal else CustomLinear(d_model,
                                                                                                            d_model,
                                                                                                            bias=True,
                                                                                                            init_eye_val=1.0))
        if use_silence:
            self.silence_linear = nn.Parameter(torch.zeros(d_model)) if bias_only else (
                CustomDiagonalLinear(d_model, bias=True, init_eye_val=non_target_rate) if is_diagonal else CustomLinear(
                    d_model, d_model, bias=True, init_eye_val=non_target_rate))

        if use_interaction:
            self.scb = SpeakerCommunicationBlock(config)

        self.use_silence = use_silence
        self.use_target = use_target
        self.use_overlap = use_overlap
        self.use_non_target = use_non_target
        self.use_interaction = use_interaction
        self.bias_only = bias_only

    def forward(self, hidden_states, stno_mask):
        stno_mask = stno_mask.to(hidden_states.device)[..., None]
        if self.bias_only:
            if self.use_silence:
                hidden_states += stno_mask[:, 0, ...] * self.silence_linear
            if self.use_target:
                hidden_states += stno_mask[:, 1, ...] * self.target_linear
            if self.use_non_target:
                hidden_states += stno_mask[:, 2, ...] * self.non_target_linear
            if self.use_overlap:
                hidden_states += stno_mask[:, 3, ...] * self.overlap_linear
        else:
            orig_hidden_states = hidden_states
            hidden_states = (self.silence_linear(
                orig_hidden_states) if self.use_silence else orig_hidden_states) * stno_mask[:, 0, :] + \
                            (self.target_linear(
                                orig_hidden_states) if self.use_target else orig_hidden_states) * stno_mask[:, 1, :] + \
                            (self.non_target_linear(
                                orig_hidden_states) if self.use_non_target else orig_hidden_states) * stno_mask[:, 2,
                                                                                                      :] + \
                            (self.overlap_linear(
                                orig_hidden_states) if self.use_overlap else orig_hidden_states) * stno_mask[:, 3, :]
        if self.use_interaction:
            hidden_states = self.scb(hidden_states)
        return hidden_states
