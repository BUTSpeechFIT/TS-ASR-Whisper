from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.utils.checkpoint
import torch.utils.checkpoint
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.speech_encoder_decoder.modeling_speech_encoder_decoder import (
    shift_tokens_right,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoder,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration,
    shift_tokens_right,
    WhisperModel,
)
from transformers.models.whisper.modeling_whisper import sinusoids
from transformers.utils import logging

from .config import Seq2SeqLMOutputLosses, Seq2SeqModelOutputLogit, DiCoWConfig
from .encoder import DiCoWEncoder
from .FDDT import FDDT
from .layers import CustomLinear, CustomDiagonalLinear, Gate, CustomLinearInitialized
from .generation import DiCoWGenerationMixin
import wandb

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


class DiCoW(WhisperModel):
    def __init__(self, config: DiCoWConfig):
        super().__init__(config)
        self.encoder = DiCoWEncoder(config)

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            stno_mask: Optional[torch.FloatTensor] = None,
            per_group_sizes: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutputLogit]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

            encoder_outputs = self.encoder(
                input_features,
                output_attentions=output_attentions,
                output_hidden_states=True,
                head_mask=head_mask,
                return_dict=return_dict,
                stno_mask=stno_mask,
                per_group_sizes=per_group_sizes
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     raise ValueError("encoder_outputs should be of type BaseModelOutput when return_dict=True.")

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.hidden_states[-1],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutputLogit(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.hidden_states[-1],
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_logits=encoder_outputs.logits,
        )


class DiCoWForConditionalGeneration(DiCoWGenerationMixin, WhisperForConditionalGeneration):
    config_class = DiCoWConfig

    def __init__(self, config: DiCoWConfig):
        super().__init__(config)
        self.model = DiCoW(config)
        self.encoder_logits = None
        self.tokenizer = None
        self.stno_mask = None
        self.stno_mask_seek = None

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _init_weights(self, module):
        std = self.config.init_std
        fddt_init = self.config.fddt_init
        if isinstance(module, CustomLinearInitialized):
            module.init_fun(module)
        elif isinstance(module, CustomLinear):
            with torch.no_grad():
                if fddt_init == 'random':
                    module.weight.data.normal_(mean=0.0, std=std)
                    if module.bias is not None:
                        module.bias.data.normal_(mean=0.0, std=std)
                elif fddt_init == 'non-disturbing':
                    module.weight.data = torch.eye(*module.weight.shape).data
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif fddt_init == 'suppressive':
                    eye = torch.eye(*module.weight.shape)
                    eye *= module.init_eye_val
                    module.weight.data = eye.data
                    if module.bias is not None:
                        module.bias.data.zero_()
        elif isinstance(module, CustomDiagonalLinear):
            with torch.no_grad():
                if fddt_init == 'random':
                    module.weight.data.normal_(mean=0.0, std=std)
                    if module.bias is not None:
                        module.bias.data.normal_(mean=0.0, std=std)
                elif fddt_init == 'non-disturbing':
                    module.weight.data = torch.ones_like(module.weight.data).data
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif fddt_init == 'suppressive':
                    module.weight.data = module.init_eye_val * torch.ones_like(module.weight.data).data
                    if module.bias is not None:
                        module.bias.data.zero_()
        elif isinstance(module, FDDT):
            if module.bias_only:
                if fddt_init == 'random':
                    module.target_linear.data.normal_(mean=0.0, std=std)
                    module.non_target_linear.data.normal_(mean=0.0, std=std)
                    module.overlap_linear.data.normal_(mean=0.0, std=std)
                    module.silence_linear.data.normal_(mean=0.0, std=std)
                    module.scb.data.normal_(mean=0.0, std=std)
                else:
                    module.target_linear.data.zero_()
                    module.non_target_linear.data.zero_()
                    module.overlap_linear.data.zero_()
                    module.silence_linear.data.zero_()
                    module.scb.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, WhisperEncoder):
            with torch.no_grad():
                embed_positions = module.embed_positions.weight
                embed_positions.copy_(sinusoids(*embed_positions.shape))
        elif isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        elif isinstance(module, nn.MultiheadAttention):
            module._reset_parameters()
        elif isinstance(module, nn.ConvTranspose1d):
            module.reset_parameters()
        elif isinstance(module, Gate):
            module.gate.data = module.init_val * torch.ones_like(module.gate.data).data

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            stno_mask: Optional[torch.FloatTensor] = None,
            per_group_sizes: Optional[torch.LongTensor] = None,
            attention_mask_enc: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            upp_labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            is_valid: Optional[bool] = None,
            spk_id: Optional[torch.LongTensor] = None,
            forced_decoder_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stno_mask=stno_mask,
            per_group_sizes=per_group_sizes
        )

        dec_lm_logits = self.proj_out(outputs.last_hidden_state)
        enc_lm_logits = outputs.encoder_logits

        loss = None
        ctc_loss = 0

        # remove fake inputs from labels and logits given per group sizes
        if is_valid is not None:
            if self.config.ctc_weight > 0.0:
                enc_lm_logits = enc_lm_logits[is_valid]
            dec_lm_logits = dec_lm_logits[is_valid]
            labels = labels[is_valid]
            upp_labels = upp_labels[is_valid]
        if labels is not None and self.config.ctc_weight > 0.0:
            enc_labels = labels.clone()
            for token in self.tokenizer.prefix_tokens:
                if (enc_labels[:, 0] == token).all():
                    enc_labels = enc_labels[:, 1:]
            enc_labels[enc_labels == self.config.eos_token_id] = -100

            ctc_loss = self.get_encoder().get_loss(enc_lm_logits, enc_labels)

        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            # move labels to correct device to enable PP
            labels = labels.to(dec_lm_logits.device)
            dec_loss1 = loss_fct(dec_lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
            dec_loss2 = loss_fct(dec_lm_logits.view(-1, self.config.vocab_size), upp_labels.reshape(-1))
            dec_loss = torch.hstack((dec_loss1[..., None], dec_loss2[..., None])).min(dim=-1).values.mean()
            if wandb.run is not None:
                wandb.log({"dec_loss": dec_loss})
                wandb.log({"ctc_loss": ctc_loss})
            loss = (1 - self.config.ctc_weight) * dec_loss + self.config.ctc_weight * ctc_loss

        if not return_dict:
            output = (dec_lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutputLosses(
            loss=loss,
            logits=dec_lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            encoder_logits=enc_lm_logits,
        )

    def _get_feat_extract_output_lengths(self, attention_mask: torch.LongTensor) -> torch.LongTensor:
        return (self.model.encoder._get_feat_extract_output_lengths(attention_mask) / 4).ceil()
