from typing import Optional, Union
import re
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import Cache
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration,
    shift_tokens_right,
    WhisperModel
)
from transformers.utils import logging
from .config import DiCoWConfig
from .encoder import DiCoWEncoder
from .generation import DiCoWGenerationMixin

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


class SoftLabelCreator(torch.nn.Module):
    """
    Handles label smoothing for timestamps and the dual-loss logic (Upper vs Lower case).
    """

    def __init__(self, tokenizer, timestamp_sigma=0.08):
        super().__init__()
        self.tokenizer = tokenizer
        self.timestamp_sigma = timestamp_sigma
        # Pre-compute the Gaussian smoothing matrix
        self.register_buffer('ts_smoothing_matrix', self._build_smoothing_matrix())

    def _build_smoothing_matrix(self):
        # FIX: Use get_vocab() instead of .decoder.items()
        vocab = self.tokenizer.get_vocab()
        vocab_size = len(vocab)

        timestamp_pattern = re.compile(r'<\|(\d+\.\d+)\|>')

        # 1. Map Token IDs to Time Values
        id_to_time = {}
        for token_str, token_id in vocab.items():
            match = timestamp_pattern.match(token_str)
            if match:
                id_to_time[token_id] = float(match.group(1))

        if not id_to_time:
            return None

        # Sorted list for fast lookups
        sorted_ids = sorted(id_to_time.keys())
        self.sorted_ts_ids = torch.tensor(sorted_ids)
        times = torch.tensor([id_to_time[i] for i in sorted_ids])

        # 2. Create the Smoothing Matrix (Num_Timestamps x Vocab_Size)
        num_ts = len(sorted_ids)
        smoothing_matrix = torch.zeros(num_ts, vocab_size)

        # Vectorized Gaussian computation
        diff_sq = (times.unsqueeze(1) - times.unsqueeze(0)) ** 2
        weights = torch.exp(-diff_sq / (2 * self.timestamp_sigma ** 2))

        # Normalize
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Scatter rows back to vocab size
        for i, ts_id in enumerate(sorted_ids):
            smoothing_matrix[i, self.sorted_ts_ids] = weights[i]

        return smoothing_matrix

    def _get_soft_distribution(self, labels, vocab_size):
        """Internal helper to convert hard labels -> soft timestamp labels"""
        device = labels.device

        # Start with One-Hot (clamp -100 to 0 temporarily)
        labels_clamped = labels.clamp(min=0)
        soft_labels = F.one_hot(labels_clamped, num_classes=vocab_size).float()

        # Apply Timestamp Smoothing if matrix exists
        if hasattr(self, 'ts_smoothing_matrix') and self.ts_smoothing_matrix is not None:
            sorted_ts_ids = self.sorted_ts_ids.to(device)
            smoothing_matrix = self.ts_smoothing_matrix.to(device)

            is_timestamp = torch.isin(labels, sorted_ts_ids)

            if is_timestamp.any():
                ts_indices = torch.searchsorted(sorted_ts_ids, labels[is_timestamp])
                soft_labels[is_timestamp] = smoothing_matrix[ts_indices]

        return soft_labels

    def compute_loss(self, logits, labels, upp_labels):
        """
        Computes the enhanced SOT loss:
        1. Generates soft labels (timestamp smoothed) for both 'labels' and 'upp_labels'.
        2. Computes KL Divergence (via CrossEntropy) for both.
        3. Takes the minimum loss per token (case invariance).
        4. Applies padding mask.
        """
        vocab_size = logits.size(-1)
        device = logits.device

        # Ensure labels are on correct device
        labels = labels.to(device)
        if upp_labels is not None:
            upp_labels = upp_labels.to(device)

        # Flatten inputs
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.reshape(-1)

        # 1. Generate Soft Targets for Lowercase
        soft_lower = self._get_soft_distribution(flat_labels, vocab_size)

        # 2. Generate Soft Targets for Uppercase (if provided)
        if upp_labels is not None:
            flat_upp = upp_labels.reshape(-1)
            soft_upper = self._get_soft_distribution(flat_upp, vocab_size)
        else:
            # Fallback if no upper labels provided (shouldn't happen in this pipeline)
            soft_upper = soft_lower

        # 3. Compute Cross Entropy (Soft Target Mode)
        # Note: CE with soft targets = -sum(target * log_prob)
        loss_fct = CrossEntropyLoss(reduction='none')

        loss_lower = loss_fct(flat_logits, soft_lower)
        loss_upper = loss_fct(flat_logits, soft_upper)

        # 4. Mask Padding (ignore_index = -100)
        # Soft-target CE doesn't support ignore_index automatically
        mask = (flat_labels != -100).float()

        loss_lower = loss_lower * mask
        loss_upper = loss_upper * mask

        # 5. Take Min (Case Invariance) and Normalize
        combined_min = torch.min(loss_lower, loss_upper)

        # Sum and divide by number of non-padding tokens
        return combined_min.sum() / mask.sum().clamp(min=1)

class DiCoW(WhisperModel):
    def __init__(self, config: DiCoWConfig):
        super().__init__(config)
        self.encoder = DiCoWEncoder(config)
        self.post_init()

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            stno_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Cache] = None,
            decoder_inputs_embeds: Optional[tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[tuple[torch.LongTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            enrollments=None
    ) -> Union[tuple[torch.Tensor], Seq2SeqModelOutput]:
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
                output_hidden_states=output_hidden_states,
                head_mask=head_mask,
                return_dict=return_dict,
                stno_mask=stno_mask,
                enrollments=enrollments
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
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
        self.soft_label_creator = None
        self.post_init()

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        # Initialize the helper class
        self.soft_label_creator = SoftLabelCreator(tokenizer)

    def get_enc_logits(self, hidden_states):
        encoder = self.model.get_encoder()
        hidden_states = encoder.possibly_update_last_hidden_states(hidden_states)
        logits = encoder.lm_head(hidden_states)
        return logits

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            stno_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Cache] = None,
            decoder_inputs_embeds: Optional[tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[tuple[torch.LongTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            upp_labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            forced_decoder_ids: Optional[torch.LongTensor] = None,
            enrollments=None,
    ) -> Union[tuple[torch.Tensor], Seq2SeqLMOutput]:

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
            cache_position=cache_position,
            stno_mask=stno_mask,
            enrollments=enrollments,
        )

        dec_lm_logits = self.proj_out(outputs.last_hidden_state)
        loss = None

        if labels is not None:
            # --- UPDATED LOSS CALCULATION ---
            if self.soft_label_creator is not None:
                # Delegate all soft label creation, flattening, and min-loss logic to the helper
                dec_loss = self.soft_label_creator.compute_loss(dec_lm_logits, labels, upp_labels)
            else:
                # Fallback to original hard label implementation if tokenizer/helper not ready
                loss_fct = CrossEntropyLoss(reduction='none')
                labels = labels.to(dec_lm_logits.device)

                flat_logits = dec_lm_logits.view(-1, self.config.vocab_size)
                dec_loss1 = loss_fct(flat_logits, labels.reshape(-1))

                if upp_labels is not None:
                    upp_labels = upp_labels.to(dec_lm_logits.device)
                    dec_loss2 = loss_fct(flat_logits, upp_labels.reshape(-1))
                    dec_loss = torch.hstack((dec_loss1[..., None], dec_loss2[..., None])).min(dim=-1).values.mean()
                else:
                    dec_loss = dec_loss1.mean()
            # --------------------------------

            if self.config.ctc_weight > 0.0:
                enc_lm_logits = self.get_enc_logits(outputs.encoder_last_hidden_state)
                # Prepare CTC labels
                enc_labels = labels.clone().to(dec_lm_logits.device)
                for token in self.tokenizer.prefix_tokens:
                    if (enc_labels[:, 0] == token).all():
                        enc_labels = enc_labels[:, 1:]
                enc_labels[enc_labels == self.config.eos_token_id] = -100

                ctc_loss = self.get_encoder().get_loss(enc_lm_logits, enc_labels)
                loss = (1 - self.config.ctc_weight) * dec_loss + self.config.ctc_weight * ctc_loss
            else:
                loss = dec_loss

        if not return_dict:
            output = (dec_lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=dec_lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def _get_feat_extract_output_lengths(self, attention_mask: torch.LongTensor) -> torch.LongTensor:
        return (self.model.get_encoder()._get_feat_extract_output_lengths(attention_mask) / 4).ceil()