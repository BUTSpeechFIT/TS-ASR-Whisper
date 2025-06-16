import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import torch.utils.checkpoint
import wandb
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
)
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor, \
    WhisperNoSpeechDetection
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from transformers.generation.utils import GenerateBeamOutput, BeamScorer, GenerateBeamDecoderOnlyOutput, \
    stack_model_outputs, GenerateBeamEncoderDecoderOutput, _split_model_inputs, GenerateNonBeamOutput, \
    GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.speech_encoder_decoder.modeling_speech_encoder_decoder import (
    shift_tokens_right,
)
from transformers.models.whisper.modeling_whisper import (
    WHISPER_ATTENTION_CLASSES,
    WhisperConfig,
    WhisperEncoder,
    WhisperEncoderLayer,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration,
    shift_tokens_right,
    WhisperModel,
)
from transformers.models.whisper.modeling_whisper import sinusoids

from mt_asr.interactions import Interaction
from utils.decoding import CTCRescorerLogitsProcessor, LogSoftmaxProcessor
from transformers.generation.configuration_utils import GenerationMode


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


class WhisperTimeStampLogitsProcessorCustom(WhisperTimeStampLogitsProcessor):
    def __init__(
            self, generate_config, begin_index: Optional[int] = None,
            _detect_timestamp_from_logprob: Optional[bool] = None
    ):  # support for the kwargs
        self.no_timestamps_token_id = generate_config.no_timestamps_token_id
        self.timestamp_begin = generate_config.no_timestamps_token_id + 1
        self.eos_token_id = generate_config.eos_token_id or generate_config.bos_token_id

        # this variable is mostly just used for testing
        self._detect_timestamp_from_logprob = (
            _detect_timestamp_from_logprob
            if _detect_timestamp_from_logprob is not None
            else getattr(generate_config, "_detect_timestamp_from_logprob", True)
        )

        num_forced_ids = (
            len(generate_config.forced_decoder_ids) if generate_config.forced_decoder_ids is not None else 0
        )
        self.begin_index = begin_index or (num_forced_ids + 1)

        self.max_initial_timestamp_index = getattr(generate_config, "max_initial_timestamp_index", None)
        self.min_initial_timestamp_index = getattr(generate_config, "min_initial_timestamp_index", None)
        # TODO(Patrick): Make sure that official models have max_initial_timestamp_index set to 50
        # self.max_initial_timestamp_index = 50

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # suppress <|notimestamps|> which is handled by without_timestamps
        scores_processed = scores.clone()
        scores_processed[:, self.no_timestamps_token_id] = -float("inf")

        # timestamps have to appear in pairs, except directly before eos_token; mask logits accordingly
        for k in range(input_ids.shape[0]):
            sampled_tokens = input_ids[k, self.begin_index:]
            seq = list(sampled_tokens.tolist())

            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    scores_processed[k, self.timestamp_begin:] = -float("inf")
                else:  # cannot be normal text tokens
                    scores_processed[k, : self.eos_token_id] = -float("inf")

            timestamps = sampled_tokens[sampled_tokens.ge(self.timestamp_begin)]
            if timestamps.numel() > 0:
                # `timestamps` shouldn't decrease; forbid timestamp tokens smaller than the last
                # The following lines of code are copied from: https://github.com/openai/whisper/pull/914/files#r1137085090
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    # Avoid to emit <|0.00|> again
                    timestamp_last = timestamps[-1] + 1

                scores_processed[k, self.timestamp_begin: timestamp_last] = -float("inf")

        # apply the `max_initial_timestamp` option
        if input_ids.shape[1] == self.begin_index:
            eos_scores = scores_processed[:, self.eos_token_id].clone()
            scores_processed[:, : self.timestamp_begin] = -float("inf")
            scores_processed[:, self.eos_token_id] = eos_scores

            if self.max_initial_timestamp_index is not None:
                last_allowed = self.timestamp_begin + self.max_initial_timestamp_index
                scores_processed[:, last_allowed + 1:] = -float("inf")
            if self.min_initial_timestamp_index is not None:
                first_allowed = self.timestamp_begin + self.min_initial_timestamp_index
                scores_processed[:, self.timestamp_begin:first_allowed] = -float("inf")

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = torch.nn.functional.log_softmax(scores_processed.float(), dim=-1)
        for k in range(input_ids.shape[0]):
            timestamp_logprob = logprobs[k, self.timestamp_begin:].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, : self.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob and self._detect_timestamp_from_logprob:
                scores_processed[k, : self.timestamp_begin] = -float("inf")

        return scores_processed


class WhisperForCTCConfig(WhisperConfig):
    """This is a modified version of the `WhisperEncoder` model from the `transformers` library.
    The model has been modified to support CTC loss computation in the forward pass."""

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
            use_target_amplifiers: bool = True,
            target_amp_is_diagonal: bool = True,
            target_amp_bias_only: bool = False,
            target_amp_use_silence: bool = True,
            target_amp_use_target: bool = True,
            target_amp_use_overlap: bool = True,
            target_amp_use_non_target: bool = True,
            remove_timestamps_from_ctc: bool = False,
            apply_target_amp_to_n_layers: int = -1,
            target_amp_init: str = 'non-disturbing',  # random, non-disturbing, dispargement
            n_soft_prompts: int = 16,
            mt_num_speakers: int = 1,
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
        self.use_target_amplifiers = use_target_amplifiers
        self.target_amp_is_diagonal = target_amp_is_diagonal
        self.target_amp_bias_only = target_amp_bias_only
        self.target_amp_use_silence = target_amp_use_silence
        self.target_amp_use_target = target_amp_use_target
        self.target_amp_use_overlap = target_amp_use_overlap
        self.target_amp_use_non_target = target_amp_use_non_target
        self.remove_timestamps_from_ctc = remove_timestamps_from_ctc
        self.apply_target_amp_to_n_layers = apply_target_amp_to_n_layers
        self.target_amp_init = target_amp_init
        self.n_soft_prompts = n_soft_prompts
        self.mt_num_speakers = mt_num_speakers


_HIDDEN_STATES_START_POSITION = 2


class CustomLinear(nn.Linear):
    def __init__(self, *args, init_eye_val=0.0, is_diagonal=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_eye_val = init_eye_val


class CustomDiagonalLinear(nn.Module):
    def __init__(self, d_model, bias=True, init_eye_val=0.0):
        super().__init__()
        self.init_eye_val = init_eye_val
        self.weight = nn.Parameter(torch.full((d_model,), init_eye_val))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, input):
        out = input * self.weight
        if self.bias is not None:
            out += self.bias
        return out


class TargetSpeakerAmplifier(nn.Module):
    def __init__(self, d_model, non_target_rate=0.01, is_diagonal=False, bias_only=False, use_silence=True,
                 use_target=True, use_overlap=True, use_non_target=True):
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

        self.use_silence = use_silence
        self.use_target = use_target
        self.use_overlap = use_overlap
        self.use_non_target = use_non_target
        self.bias_only = bias_only

    def forward(self, hidden_states, vad_mask):
        vad_mask = vad_mask.to(hidden_states.device)[..., None]
        if self.bias_only:
            if self.use_target:
                hidden_states += vad_mask[:, 1, ...] * self.target_linear
        return hidden_states


class ResMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn_dim = dim // 2
        self.net = nn.Sequential(
            nn.Linear(dim, self.bn_dim),
            nn.ReLU(),
            nn.Linear(self.bn_dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return x + self.net(x)


class WhisperEncoderForCTC(WhisperEncoder):
    config_class = WhisperForCTCConfig

    def __init__(self, config: WhisperForCTCConfig):
        super().__init__(config)
        if config.additional_layer:
            self.additional_layer = WhisperEncoderLayer(config)
        if config.additional_self_attention_layer:
            self.additional_self_attention_layer = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
                embed_dim=config.d_model,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config,
            )
        self.ctc_weight = config.ctc_weight
        if config.sub_sample:
            self.subsample_conv1 = nn.Conv1d(
                in_channels=config.d_model,
                out_channels=config.d_model,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            self.subsample_conv2 = nn.Conv1d(
                in_channels=config.d_model,
                out_channels=config.d_model,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size + 1, bias=False)
        self.final_dropout = nn.Dropout(config.final_dropout)
        if config.use_target_amplifiers:
            num_amplifiers = self.config.apply_target_amp_to_n_layers if self.config.apply_target_amp_to_n_layers != -1 else len(
                self.layers)
            self.target_amplifiers = nn.ModuleList([
                TargetSpeakerAmplifier(config.d_model,
                                       non_target_rate=0.0 if i == 0 else 1.0,
                                       is_diagonal=config.target_amp_is_diagonal,
                                       bias_only=config.target_amp_bias_only,
                                       use_silence=config.target_amp_use_silence,
                                       use_target=config.target_amp_use_target,
                                       use_overlap=config.target_amp_use_overlap,
                                       use_non_target=config.target_amp_use_non_target)

                for i in range(num_amplifiers)
            ])
        self.first_timestamp_position = self.config.vocab_size - 30 * 50  # 30 seconds of 50 Hz timestamps
        if config.mt_num_speakers > 1:
            self.interaction = Interaction(config)
        self.post_init()

    def _init_weights(self, module):
        std = self.config.init_std
        target_amp_init_method = self.config.target_amp_init
        if isinstance(module, CustomLinear):
            with torch.no_grad():
                if target_amp_init_method == 'random':
                    module.weight.data.normal_(mean=0.0, std=std)
                    if module.bias is not None:
                        module.bias.data.normal_(mean=0.0, std=std)
                elif target_amp_init_method == 'non-disturbing':
                    module.weight.data = torch.eye(*module.weight.shape).data
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif target_amp_init_method == 'disparagement':
                    eye = torch.eye(*module.weight.shape)
                    eye *= module.init_eye_val
                    module.weight.data = eye.data
                    if module.bias is not None:
                        module.bias.data.zero_()
        elif isinstance(module, CustomDiagonalLinear):
            with torch.no_grad():
                if target_amp_init_method == 'random':
                    module.weight.data.normal_(mean=0.0, std=std)
                    if module.bias is not None:
                        module.bias.data.normal_(mean=0.0, std=std)
                elif target_amp_init_method == 'non-disturbing':
                    module.weight.data = torch.ones_like(module.weight.data).data
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif target_amp_init_method == 'disparagement':
                    module.weight.data = module.init_eye_val * torch.ones_like(module.weight.data).data
                    if module.bias is not None:
                        module.bias.data.zero_()
        elif isinstance(module, TargetSpeakerAmplifier):
            if module.bias_only:
                if target_amp_init_method == 'random':
                    module.target_linear.data.normal_(mean=0.0, std=std)
                    module.non_target_linear.data.normal_(mean=0.0, std=std)
                    module.overlap_linear.data.normal_(mean=0.0, std=std)
                    module.silence_linear.data.normal_(mean=0.0, std=std)
                else:
                    module.target_linear.data.zero_()
                    module.non_target_linear.data.zero_()
                    module.overlap_linear.data.zero_()
                    module.silence_linear.data.zero_()
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

    @classmethod
    def _load_pretrained_model(
            cls,
            model,
            state_dict,
            loaded_keys,
            resolved_archive_file,
            pretrained_model_name_or_path,
            **kwargs
    ):
        for key in list(state_dict.keys()):
            if key.startswith("encoder."):
                state_dict[key[8:]] = state_dict.pop(key)
                loaded_keys.remove(key)
                loaded_keys.append(key[8:])
        output = super()._load_pretrained_model(
            model,
            state_dict,
            loaded_keys,
            resolved_archive_file,
            pretrained_model_name_or_path,
            **kwargs
        )
        return output

    def get_loss(self, logits, labels):
        if labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")
        if self.config.remove_timestamps_from_ctc:
            labels = torch.nn.utils.rnn.pad_sequence([label[label < self.first_timestamp_position] for label in labels],
                                                     padding_value=-100).T
        input_lengths = torch.full((logits.shape[0],), fill_value=logits.shape[1],
                                   device=logits.device)

        # assuming that padded tokens are filled with -100
        # when not being attended to
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        # flattened_targets = labels_enc.masked_select(labels_mask)

        # ctc_loss doesn't support fp16
        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

        with torch.backends.cudnn.flags(enabled=True):
            ctc_loss = nn.functional.ctc_loss(
                log_probs,
                labels,
                input_lengths,
                target_lengths,
                blank=logits.shape[-1] - 1,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=True,
            )
        return ctc_loss

    def forward(
            self,
            input_features,
            attention_mask=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            vad_mask=None,
            per_group_sizes=None
    ):
        # For MT-ASR the input has shape (B X S) x F x T
        # we can use torch.view(B, S, F, -1) to obtain
        # new tensor with speaker dim
        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            if input_features.shape[-1] > expected_seq_length:
                return CausalLMOutput(
                    logits=None,
                    hidden_states=None,
                    attentions=None,
                )
            else:
                raise ValueError(
                    f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight
        if hasattr(self, "shift_embeds") and self.shift_embeds:
            embed_pos = embed_pos[
                torch.clamp(((vad_mask[:, 1, :] + vad_mask[:, 3, :]).cumsum(dim=-1) - 1), min=0).to(torch.long)]

        hidden_states = inputs_embeds + embed_pos

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if self.config.use_target_amplifiers and idx < len(self.target_amplifiers):
                hidden_states = self.target_amplifiers[idx](hidden_states, vad_mask)

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            outputs = tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        else:
            outputs = BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            )

        if hasattr(self, "interaction"):
            outputs.last_hidden_state = self.interaction(
                outputs.last_hidden_state,
                per_group_sizes
            )
            outputs.hidden_states = (*outputs.hidden_states[:-1], outputs.last_hidden_state)

        if self.config.additional_layer:
            inter_output, = self.additional_layer(
                outputs.last_hidden_state,
                attention_mask=None,
                output_attentions=output_attentions,
                layer_head_mask=None,
            )
        elif self.config.additional_self_attention_layer:
            inter_output, _, __ = self.additional_self_attention_layer(
                outputs.last_hidden_state,
                attention_mask=None,
                output_attentions=output_attentions,
                layer_head_mask=None,
            )
        else:
            inter_output = outputs.last_hidden_state

        inter_output = self.final_dropout(inter_output)
        if self.config.sub_sample:
            inter_output = self.subsample_conv2(self.subsample_conv1(inter_output.transpose(1, 2))).transpose(1, 2)
        logits = self.lm_head(inter_output)

        return CausalLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class WhisperModelWithCTC(WhisperModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = WhisperEncoderForCTC(config)

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
            vad_mask: Optional[torch.FloatTensor] = None,
            per_group_sizes: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutputLosses]:
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> from transformers import AutoFeatureExtractor, WhisperModel
         >>> from datasets import load_dataset

         >>> model = WhisperModel.from_pretrained("openai/whisper-base")
         >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
         >>> input_features = inputs.input_features
         >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
         >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
         >>> list(last_hidden_state.shape)
         [1, 2, 512]
         ```"""
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
                vad_mask=vad_mask,
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


class WhisperForConditionalGenerationWithCTC(WhisperForConditionalGeneration):
    config_class = WhisperForCTCConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = WhisperModelWithCTC(config)
        self.encoder_logits = None
        self.tokenizer = None
        self.vad_seek_callback = None

    # We need this setter as we can't pass a function/method as a config argument.
    # JSON serialization fails at that point.
    def set_vad_seek_callback(self, vad_seek_callback):
        self.vad_seek_callback = vad_seek_callback

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _init_weights(self, module):
        std = self.config.init_std
        target_amp_init_method = self.config.target_amp_init
        if isinstance(module, CustomLinear):
            with torch.no_grad():
                if target_amp_init_method == 'random':
                    module.weight.data.normal_(mean=0.0, std=std)
                    if module.bias is not None:
                        module.bias.data.normal_(mean=0.0, std=std)
                elif target_amp_init_method == 'non-disturbing':
                    module.weight.data = torch.eye(*module.weight.shape).data
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif target_amp_init_method == 'disparagement':
                    eye = torch.eye(*module.weight.shape)
                    eye *= module.init_eye_val
                    module.weight.data = eye.data
                    if module.bias is not None:
                        module.bias.data.zero_()
        elif isinstance(module, CustomDiagonalLinear):
            with torch.no_grad():
                if target_amp_init_method == 'random':
                    module.weight.data.normal_(mean=0.0, std=std)
                    if module.bias is not None:
                        module.bias.data.normal_(mean=0.0, std=std)
                elif target_amp_init_method == 'non-disturbing':
                    module.weight.data = torch.ones_like(module.weight.data).data
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif target_amp_init_method == 'disparagement':
                    module.weight.data = module.init_eye_val * torch.ones_like(module.weight.data).data
                    if module.bias is not None:
                        module.bias.data.zero_()
        elif isinstance(module, TargetSpeakerAmplifier):
            if module.bias_only:
                if target_amp_init_method == 'random':
                    module.target_linear.data.normal_(mean=0.0, std=std)
                    module.non_target_linear.data.normal_(mean=0.0, std=std)
                    module.overlap_linear.data.normal_(mean=0.0, std=std)
                    module.silence_linear.data.normal_(mean=0.0, std=std)
                else:
                    module.target_linear.data.zero_()
                    module.non_target_linear.data.zero_()
                    module.overlap_linear.data.zero_()
                    module.silence_linear.data.zero_()
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

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            vad_mask: Optional[torch.FloatTensor] = None,
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
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
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
            vad_mask=vad_mask,
            per_group_sizes=per_group_sizes
        )

        dec_lm_logits = self.proj_out(outputs.last_hidden_state)
        enc_lm_logits = outputs.encoder_logits

        loss = None
        ctc_loss = 0
        if labels is not None and self.ctc_weight > 0.0:
            enc_labels = labels.clone()
            for token in self.tokenizer.prefix_tokens:
                if (enc_labels[:, 0] == token).all():
                    enc_labels = enc_labels[:, 1:]
            enc_labels[enc_labels == self.config.eos_token_id] = -100

            ctc_loss = self.get_encoder().get_loss(enc_lm_logits, enc_labels)
            if wandb.run is not None:
                wandb.log({"ctc_loss": ctc_loss})

        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            # move labels to correct device to enable PP
            labels = labels.to(dec_lm_logits.device)
            dec_loss1 = loss_fct(dec_lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
            dec_loss2 = loss_fct(dec_lm_logits.view(-1, self.config.vocab_size), upp_labels.reshape(-1))
            dec_loss = torch.hstack((dec_loss1[..., None], dec_loss2[..., None])).min(dim=-1).values.mean()
            loss = (1 - self.ctc_weight) * dec_loss + self.ctc_weight * ctc_loss

            if wandb.run is not None:
                wandb.log({"att_loss": dec_loss})

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

    def _get_feat_extract_output_lengths(self, attention_mask: torch.Tensor) -> torch.Tensor:
        return (self.model.encoder._get_feat_extract_output_lengths(attention_mask) / 4).ceil()

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name, generation_config,
    ) -> Dict[str, Any]:
        # self.encoder_output_lens = self._get_feat_extract_output_lengths(
        #     model_kwargs['attention_mask_enc'].sum(dim=1)
        # ).int()
        generation_config.output_hidden_states = True

        # pylint: disable=no-memberva
        model_kwargs = super()._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )
        self.encoder_logits = model_kwargs["encoder_outputs"].logits

        return model_kwargs

    def _get_logits_processor(
            self,
            generation_config: GenerationConfig,
            input_ids_seq_length: int,
            encoder_input_ids: torch.LongTensor,
            prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
            logits_processor: Optional[LogitsProcessorList],
            device: str = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorList:
        # pylint: disable=no-member
        processors = super()._get_logits_processor(
            generation_config,
            input_ids_seq_length,
            encoder_input_ids,
            prefix_allowed_tokens_fn,
            logits_processor,
            device,
            model_kwargs,
            negative_prompt_ids,
            negative_prompt_attention_mask,
        )
        if hasattr(generation_config, "ctc_weight") and generation_config.ctc_weight > 0:
            enc_logits = self.encoder_logits
            if generation_config.num_beams <= 1:
                processors.append(LogSoftmaxProcessor())
            else:
                enc_logits = enc_logits.repeat_interleave(generation_config.num_beams, dim=0)
            self.ctc_rescorer = CTCRescorerLogitsProcessor(
                enc_logits,
                torch.full((enc_logits.shape[0],), fill_value=enc_logits.shape[1],
                           device=enc_logits.device),
                enc_logits.shape[-1] - 1,
                generation_config.pad_token_id.item(),
                generation_config.eos_token_id.item(),
                generation_config.decoder_start_token_id.item(),
                self.tokenizer,
                generation_config.ctc_margin,
                generation_config.ctc_weight,
                generation_config.num_beams,
                False,
            )
            processors.append(self.ctc_rescorer)
        return processors

    @staticmethod
    def _expand_inputs_for_generation(
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            input_ids: Optional[torch.LongTensor] = None,
            **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor) and key != "loss":
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])
            if "hidden_states" in model_kwargs["encoder_outputs"]:
                model_kwargs["encoder_outputs"]["hidden_states"] = tuple(
                    hidden_state.repeat_interleave(expand_size, dim=0) for hidden_state in
                    model_kwargs["encoder_outputs"]["hidden_states"]
                )

        return input_ids, model_kwargs

    def _retrieve_logit_processors(self, generation_config, logits_processor, begin_index, is_shortform, num_beams):
        if generation_config.return_timestamps is True:
            timestamp_processor = WhisperTimeStampLogitsProcessorCustom(generation_config, begin_index=begin_index)
            logits_processor = (
                [timestamp_processor] if logits_processor is None else [timestamp_processor] + logits_processor
            )

        if generation_config.suppress_tokens is not None:
            suppress_tokens_processor = SuppressTokensLogitsProcessor(generation_config.suppress_tokens)
            logits_processor = (
                [suppress_tokens_processor]
                if logits_processor is None
                else [suppress_tokens_processor] + logits_processor
            )
            generation_config.suppress_tokens = None

        if generation_config.begin_suppress_tokens is not None:
            begin_suppress_processor = SuppressTokensAtBeginLogitsProcessor(
                generation_config.begin_suppress_tokens, begin_index=begin_index
            )
            logits_processor = (
                [begin_suppress_processor]
                if logits_processor is None
                else [begin_suppress_processor] + logits_processor
            )
            generation_config.begin_suppress_tokens = None

        if generation_config.no_speech_threshold is not None and not is_shortform:
            no_speech_detector = WhisperNoSpeechDetection(
                no_speech_token=generation_config.no_timestamps_token_id - 1,
                begin_index=begin_index,
                scores_is_logprobs=num_beams > 1,
            )
            logits_processor = (
                [no_speech_detector] if logits_processor is None else [no_speech_detector] + logits_processor
            )
            no_speech_detector.set_model(self)

        return logits_processor

    def generate(
            self,
            input_features: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: bool = False,
            return_timestamps: Optional[bool] = None,
            task: Optional[str] = None,
            language: Optional[str] = None,
            is_multilingual: Optional[bool] = None,
            prompt_ids: Optional[torch.Tensor] = None,
            prompt_condition_type: Optional[str] = None,  # first-segment, all-segments
            condition_on_prev_tokens: Optional[bool] = None,
            temperature: Optional[Union[float, Tuple[float, ...]]] = None,
            compression_ratio_threshold: Optional[float] = None,
            logprob_threshold: Optional[float] = None,
            no_speech_threshold: Optional[float] = None,
            num_segment_frames: Optional[int] = None,
            attention_mask: Optional[torch.Tensor] = None,
            time_precision: float = 0.02,
            return_token_timestamps: Optional[bool] = None,
            return_segments: bool = False,
            return_dict_in_generate: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            **kwargs,
    ):

        gen_c, _ = self._prepare_generation_config(generation_config, **kwargs)
        gen_mode = gen_c.get_generation_mode(assistant_model)

        if gen_mode not in [GenerationMode.GREEDY_SEARCH, GenerationMode.BEAM_SEARCH]:
            raise ValueError(
                f"Provided generation mode {gen_mode} is not supported"
                f" for WhisperForConditionalGeneration with joint CTC decoding")

        if "vad_mask" in kwargs:
            self.vad_mask = kwargs["vad_mask"]
        if "encoder_outputs" in kwargs:
            self.encoder_logits = kwargs["encoder_outputs"].logits
        # pylint: disable=no-member
        output = super().generate(
            input_features=input_features,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            return_timestamps=self.generation_config.return_timestamps if hasattr(self.generation_config,
                                                                                  "return_timestamps") else return_timestamps,
            task=task,
            language=language,
            is_multilingual=is_multilingual,
            prompt_ids=prompt_ids,
            prompt_condition_type=prompt_condition_type,
            condition_on_prev_tokens=condition_on_prev_tokens,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            num_segment_frames=num_segment_frames,
            attention_mask=attention_mask,
            time_precision=time_precision,
            return_token_timestamps=return_token_timestamps,
            return_segments=return_segments,
            return_dict_in_generate=return_dict_in_generate,
            **kwargs,
        )
        self.encoder_logits = None
        return output

    @staticmethod
    def _retrieve_segment(
            seek_sequence,
            seek_outputs,
            time_offset,
            timestamp_begin,
            seek_num_frames,
            time_precision,
            input_stride,
            prev_idx,
            idx,
            return_token_timestamps,
    ):
        # find the predicted "end of segment" predictions of Whisper
        # "end of segment" predictions occur whenever Whisper predicts a timestamp token
        timestamp_tokens: torch.Tensor = seek_sequence.ge(timestamp_begin)
        single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]
        timestamp_segment_indices = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
        timestamp_segment_indices.add_(1)
        token_timestamps = seek_outputs[idx]["token_timestamps"] if return_token_timestamps else []

        # If whisper predicted a "end of segment" via a timestep token, let's go ever each
        # "end of segment" prediction and slice the decoding into segments accordingly
        if len(timestamp_segment_indices) > 0:
            # if the output contains two consecutive timestamp tokens
            slices = timestamp_segment_indices.tolist()
            segments = []
            if single_timestamp_ending:
                slices.append(len(seek_sequence))

            last_slice = 0
            # Add each segment to list of all segments
            for current_slice in slices:
                sliced_tokens = seek_sequence[last_slice:current_slice]
                start_timestamp_pos = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_pos = sliced_tokens[-1].item() - timestamp_begin
                segments.append(
                    {
                        "start": time_offset[prev_idx] + start_timestamp_pos * time_precision,
                        "end": time_offset[prev_idx] + end_timestamp_pos * time_precision,
                        "tokens": sliced_tokens,
                        "result": seek_outputs[idx],
                    }
                )
                if return_token_timestamps:
                    segments[-1]["token_timestamps"] = (
                            token_timestamps[last_slice:current_slice] + time_offset[prev_idx]
                    )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                segment_offset = seek_num_frames[prev_idx]
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                # here we throw away all predictions after the last predicted "end of segment"
                # since we are cutting right in the middle of an audio
                last_timestamp_pos = seek_sequence[last_slice - 1].item() - timestamp_begin
                segment_offset = last_timestamp_pos * input_stride
        else:
            # If whisper does not predict any "end of segment" token, then
            # the whole decoding is considered a segment and we add it to the list of segments
            timestamps = seek_sequence[timestamp_tokens.nonzero().flatten()]
            last_timestamp_pos = seek_num_frames[prev_idx]
            if timestamps.numel() > 0 and timestamps[-1].item() != timestamp_begin:
                # no consecutive timestamps but it has a timestamp; use the last one.
                last_timestamp_pos = timestamps[-1].item() - timestamp_begin
            start_timestamp = (timestamps[0].item() - timestamp_begin) * time_precision
            segments = [
                {
                    "start": time_offset[prev_idx] + start_timestamp,
                    "end": time_offset[prev_idx] + last_timestamp_pos * time_precision,
                    "tokens": seek_sequence,
                    "result": seek_outputs[idx],
                }
            ]
            if return_token_timestamps:
                segments[-1]["token_timestamps"] = token_timestamps + time_offset[prev_idx]
            segment_offset = seek_num_frames[prev_idx]

        return segments, segment_offset

    def _postprocess_outputs(self, seek_outputs, decoder_input_ids, return_token_timestamps, generation_config):
        # remove all previously passed decoder input ids
        if isinstance(seek_outputs, torch.Tensor):
            seek_outputs = seek_outputs[:, decoder_input_ids.shape[-1]:]
            seek_outputs = torch.hstack((seek_outputs, torch.full((seek_outputs.shape[0], 2),
                                                                  fill_value=generation_config.eos_token_id,
                                                                  dtype=seek_outputs.dtype,
                                                                  device=seek_outputs.device)))
            first_eos = (seek_outputs == generation_config.eos_token_id).int().argmax(dim=1)
            biggest_timestamp = generation_config.no_timestamps_token_id + 1 + 30 * 50

            empty_transcriptions = first_eos == 0
            seek_outputs[empty_transcriptions, 0] = generation_config.no_timestamps_token_id + 1  # 0.00 timestamp
            seek_outputs[empty_transcriptions, 1] = generation_config.bos_token_id
            seek_outputs[empty_transcriptions, 2] = biggest_timestamp  # 30.00 timestamp

            first_eos = (seek_outputs == generation_config.eos_token_id).int().argmax(dim=1)
            seek_outputs[torch.arange(seek_outputs.shape[0]), first_eos] = biggest_timestamp
            seek_outputs[torch.arange(seek_outputs.shape[0]), first_eos + 1 * (
                    first_eos < seek_outputs.shape[1] - 1)] = biggest_timestamp

            return seek_outputs, seek_outputs

        if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
            num_frames = getattr(generation_config, "num_frames", None)
            seek_outputs["token_timestamps"] = self._extract_token_timestamps(
                seek_outputs, generation_config.alignment_heads, num_frames=num_frames
            )
            seek_outputs["token_timestamps"] = seek_outputs["token_timestamps"][:, decoder_input_ids.shape[-1]:]

        seek_outputs["sequences"] = seek_outputs["sequences"][:, decoder_input_ids.shape[-1]:]

        def split_by_batch_index(values, key, batch_idx):
            if key == "scores":
                return [v[batch_idx].cpu() for v in values]
            elif key == "past_key_values":
                # we don't save `past_key_values` as this is too costly
                return None
            elif isinstance(values[batch_idx], tuple) and torch.is_tensor(values[batch_idx][0]):
                return tuple(tuple(w[batch_idx][None].cpu() for w in v) for v in values)
            return values[batch_idx].cpu()

        sequence_tokens = seek_outputs["sequences"]
        seek_outputs = [
            {k: split_by_batch_index(v, k, i) for k, v in seek_outputs.items()}
            for i in range(sequence_tokens.shape[0])
        ]

        return sequence_tokens, seek_outputs

    def generate_with_fallback(
            self,
            segment_input,
            decoder_input_ids,
            cur_bsz,
            batch_idx_map,
            seek,
            num_segment_frames,
            max_frames,
            temperatures,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            return_token_timestamps,
            do_condition_on_prev_tokens,
            kwargs,
    ):
        kwargs = copy.copy(kwargs)
        kwargs["attention_mask_enc"] = torch.ones(cur_bsz, segment_input.size(-1), device=segment_input.device)
        seek_vad = seek // 2
        num_frames_vad = num_segment_frames // 2
        max_frames_vad = max_frames // 2
        seek_num_frames = (max_frames_vad - seek_vad).clamp(max=num_frames_vad)

        vad_masks = []
        for i in range(cur_bsz):
            prev_i = batch_idx_map[i]
            segment_input_slice = kwargs["vad_mask"][prev_i: prev_i + 1, :,
                                  seek_vad[prev_i]: seek_vad[prev_i] + seek_num_frames[prev_i]]

            if segment_input_slice.shape[-1] < num_frames_vad:
                orig_len = segment_input_slice.shape[-1]
                # pad to 3000 if necessary
                segment_input_slice = torch.nn.functional.pad(
                    segment_input_slice, pad=(0, num_frames_vad - orig_len)
                )
                # set corresponding padding tokens to 1 in vad mask representing silence
                segment_input_slice[0, 0, orig_len:] = 1.0

            vad_masks.append(segment_input_slice)
        kwargs["vad_mask"] = torch.cat(vad_masks, dim=0)

        if "per_group_sizes" in kwargs:
            group_sizes = kwargs["per_group_sizes"].clone()
            group_sizes[:] = 0
            cummulative_group_sizes = kwargs["per_group_sizes"].cumsum(dim=0)
            for i in batch_idx_map:
                group_idx = (cummulative_group_sizes > i).nonzero().min()
                group_sizes[group_idx] += 1
            kwargs["per_group_sizes"] = group_sizes

        if self.vad_seek_callback is not None:
            self.vad_seek_callback(kwargs["vad_mask"])

        # 6.6 Batch generate current chunk
        seek_sequence_list = [None for _ in range(cur_bsz)]
        seek_outputs_list = [None for _ in range(cur_bsz)]
        needs_fallback = [False for _ in range(cur_bsz)]
        should_skip = [False for _ in range(cur_bsz)]
        fallback_index_map = list(range(cur_bsz))

        if generation_config.no_speech_threshold is not None:
            self._setup_no_speech_detection(logits_processor, segment_input, decoder_input_ids, kwargs)

        for fallback_idx, temperature in enumerate(temperatures):
            generation_config.do_sample = temperature is not None and temperature > 0.0
            generation_config.temperature = temperature if generation_config.do_sample else 1.0
            if generation_config.do_sample:
                generation_config.num_beams = 1

            generate_kwargs = copy.copy(kwargs)

            for key in ["do_sample", "temperature", "num_beams"]:
                if key in generate_kwargs:
                    del generate_kwargs[key]

            seek_outputs = super().generate(
                segment_input,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                decoder_input_ids=decoder_input_ids,
                **generate_kwargs,
            )

            # post-process sequence tokens and outputs to be in list form
            seek_sequences, seek_outputs = self._postprocess_outputs(
                seek_outputs=seek_outputs,
                decoder_input_ids=decoder_input_ids,
                return_token_timestamps=return_token_timestamps,
                generation_config=generation_config,
            )

            # 6.7 Extract cut sequences from every sequence and check if fallback should be applied
            # Loop over each decoded audio individually as each decoding can be of a different length
            new_fallback_index_map = []
            new_segment_input = []
            new_decoder_input_ids = []
            new_decoder_attention_mask = []

            for i, seek_sequence in enumerate(seek_sequences):
                # make sure we cut a predicted EOS token if we are not finished with the generation yet
                prev_i = batch_idx_map[fallback_index_map[i]]
                is_not_final = (seek[prev_i] + num_segment_frames) < max_frames[prev_i]

                # remove eos token id
                if is_not_final and seek_sequence[-1] == generation_config.eos_token_id:
                    seek_sequence = seek_sequence[:-1]
                    if return_token_timestamps:
                        seek_outputs[i]["token_timestamps"] = seek_outputs[i]["token_timestamps"][:-1]

                # remove all padding tokens
                if seek_sequence[-1] == generation_config.pad_token_id:
                    num_paddings = (seek_sequence == generation_config.pad_token_id).sum()
                    seek_sequence = seek_sequence[:-num_paddings]
                    if return_token_timestamps:
                        seek_outputs[i]["token_timestamps"] = seek_outputs[i]["token_timestamps"][:-num_paddings]

                # check which sequences in batch need fallback & which should be skipped
                needs_fallback[i], should_skip[i] = self._need_fallback(
                    seek_sequence,
                    seek_outputs,
                    i,
                    logits_processor,
                    generation_config,
                    self.config.vocab_size,
                    temperature,
                )

                seek_sequence_list[fallback_index_map[i]] = seek_sequence
                seek_outputs_list[fallback_index_map[i]] = seek_outputs[i]
                is_low_temperature = temperature is None or temperature < 0.5
                do_condition_on_prev_tokens[fallback_index_map[i]] = (
                        generation_config.condition_on_prev_tokens and is_low_temperature
                )

                if needs_fallback[i]:
                    new_fallback_index_map.append(fallback_index_map[i])
                    new_segment_input.append(segment_input[i])
                    new_decoder_input_ids.append(decoder_input_ids[i])
                    if "decoder_attention_mask" in kwargs:
                        new_decoder_attention_mask.append(kwargs["decoder_attention_mask"][i])

            fallback_index_map = new_fallback_index_map

            # if no sequence needs to be run with temperature fallback, we're finished
            if len(fallback_index_map) == 0 or fallback_idx == len(temperatures) - 1:
                seek_sequences = seek_sequence_list
                seek_outputs = seek_outputs_list
                break

            # if we're still in the loop, make sure that decoder_input_ids and segment inputs are tensors
            decoder_input_ids = torch.stack(new_decoder_input_ids)
            segment_input = torch.stack(new_segment_input)
            if "decoder_attention_mask" in kwargs:
                kwargs["decoder_attention_mask"] = torch.stack(new_decoder_attention_mask)

        return seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens

    def freeze_except(self, prefixes_to_preheat):
        for name, param in self.named_parameters():
            param.requires_grad = False
            for prefix in prefixes_to_preheat:
                if name.startswith(prefix):
                    param.requires_grad = True

    def suppress_interactions(self):
        """This method suppress final projection in CoAttention blocks to let the original information flow through"""
        for name, param in self.named_parameters():
            if "interaction" in name and "cat_proj" in name:
                with torch.no_grad():
                    if "bias" in name:
                        param[:] = 0.
                    else:
                        param[:] *= 0.001

    def _beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamScorer,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool,
            logits_warper: Optional[LogitsProcessorList] = None,
            **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`:
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # if sequential is True, split the input to batches of batch_size and run sequentially
            if sequential:
                if any(
                        model_name in self.__class__.__name__.lower()
                        for model_name in [
                            "fsmt",
                            "reformer",
                            "bloom",
                            "ctrl",
                            "gpt_bigcode",
                            "transo_xl",
                            "xlnet",
                            "cpm",
                            "jamba",
                        ]
                ):
                    raise RuntimeError(
                        f"Currently generation for {self.__class__.__name__} is not supported "
                        f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                    )

                inputs_per_sub_batches = _split_model_inputs(
                    model_inputs, split_size=batch_size, full_batch_size=batch_beam_size
                )
                outputs_per_sub_batch = [
                    self(
                        **inputs_per_sub_batch,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                    for inputs_per_sub_batch in inputs_per_sub_batches
                ]

                outputs = stack_model_outputs(outputs_per_sub_batch)

            else:  # Unchanged original behavior
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            if do_sample:
                next_token_scores_processed = logits_warper(input_ids, next_token_scores_processed)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
            # non eos token per beam.
            n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
            n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)
            else:
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
                )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            """ 
            
            
            
            Based on the beam idx and next tokens reshuffle the ctc prev states and scores
            
            
            
            
            """
            if hasattr(self, "ctc_rescorer"):
                self.ctc_rescorer.update_state(beam_next_tokens, beam_idx)
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]

    def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool,
            streamer: Optional["BaseStreamer"],
            logits_warper: Optional[LogitsProcessorList] = None,
            **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config.pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            """ 



            Based on the next tokens select the ctc prev states and scores



            """
            if hasattr(self, "ctc_rescorer"):
                self.ctc_rescorer.update_state(next_tokens, torch.arange(next_tokens.shape[0]))

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
