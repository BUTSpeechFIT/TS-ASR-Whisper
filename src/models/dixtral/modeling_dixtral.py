# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import math
from typing import Callable, Optional, Union

import wandb
import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.generic import check_model_inputs
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from .configuration_dixtral import DixtralConfig, DixtralEncoderConfig
from transformers.models.voxtral import  VoxtralConfig
from transformers.models.llama.modeling_llama import  LlamaDecoderLayer

from src.models.dicow.FDDT import FDDT
from src.models.dicow.layers import CustomLinear, CustomDiagonalLinear


logger = logging.get_logger(__name__)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None and attention_mask.ndim == 4:
        attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.view(1, -1, 1, 1)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class VoxtralAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        layer_idx: Optional[int] = None,
        config: Optional[VoxtralConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        if layer_idx is None and is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()

        # Scaling is susceptible to floating point arithmetics' inprecisions
        # which can lead to different results (this is dependent from model
        # to model, e.g. whisper is one such case). We therefore keep the
        # original order of scaling to follow the original implementation
        # and enforce no scaling (1.0) in the attention call below.
        query_states = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            output_attentions=output_attentions,
            head_mask=layer_head_mask,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class VoxtralEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: VoxtralConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = VoxtralAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states, attn_weights


@auto_docstring
class DixtralPreTrainedModel(PreTrainedModel):
    config: DixtralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = None
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_attention_backend = True
    _can_compile_fullgraph = True

    def _init_weights(self, module):
        # important: this ported version of Voxtral isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.audio_config.initializer_range
        )

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (CustomLinear, CustomDiagonalLinear)):
            module.reset_parameters()


@auto_docstring(
    custom_intro="""
    The Voxtral encoder, which is a Whisper encoder.
    """
)
class DixtralEncoder(DixtralPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`VoxtralEncoderLayer`].

    Args:
        config: VoxtralEncoderConfig
    """

    # Ignore copy
    config: DixtralEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["VoxtralEncoderLayer"]
    _can_record_outputs = {
        "attentions": VoxtralAttention,
        "hidden_states": VoxtralEncoderLayer,
    }

    def __init__(self, config: DixtralEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        self.layers = nn.ModuleList([VoxtralEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

        self._init_dicow_components(config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _init_dicow_components(self, config):
        """Initialize DiCoW-specific components"""
        if not config.use_dicow_encoder:
            return

        # FDDT components
        if config.use_fddt:
            num_fddts = (config.apply_fddt_to_n_layers
                        if config.apply_fddt_to_n_layers != -1
                        else len(self.layers))
            self.fddts = nn.ModuleList([
                FDDT(
                    d_model=config.d_model,
                    non_target_rate=1.0,
                    fddt_init=config.fddt_init,
                    is_diagonal=config.fddt_is_diagonal,
                    bias_only=config.fddt_bias_only,
                    use_silence=config.fddt_use_silence,
                    use_target=config.fddt_use_target,
                    use_overlap=config.fddt_use_overlap,
                    use_non_target=config.fddt_use_non_target,
                )
                for _ in range(num_fddts)
            ])

            if config.use_pre_pos_fddt:
                self.initial_fddt = FDDT(
                    d_model=config.d_model,
                    non_target_rate=config.non_target_fddt_value,
                    fddt_init=config.fddt_init,
                    is_diagonal=config.fddt_is_diagonal,
                    bias_only=config.fddt_bias_only,
                    use_silence=config.fddt_use_silence,
                    use_target=config.fddt_use_target,
                    use_overlap=config.fddt_use_overlap,
                    use_non_target=config.fddt_use_non_target,
                )

        # For CTC label processing
        self.first_task_token = config.vocab_size - 30 * 50 - 1 - 6

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value


    @check_model_inputs
    def forward(
        self,
        input_features,
        attention_mask=None,
        stno_mask=None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Voxtral does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
        """
        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Qwen2Audio expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        # Apply initial FDDT if configured
        if (self.config.use_dicow_encoder and
                self.config.use_fddt and
                self.config.use_pre_pos_fddt and
                hasattr(self, 'initial_fddt')):
            inputs_embeds = self.initial_fddt(inputs_embeds, stno_mask)

        embed_pos = self.embed_positions.weight
        hidden_states = (inputs_embeds + embed_pos).to(inputs_embeds.dtype)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        for idx, encoder_layer in enumerate(self.layers):

            if (self.config.use_dicow_encoder and
                self.config.use_fddt and
                hasattr(self, 'fddts') and
                idx < len(self.fddts)):
                hidden_states = self.fddts[idx](hidden_states, stno_mask)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=None,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class VoxtralMultiModalProjector(nn.Module):
    def __init__(self, config: VoxtralConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.audio_config.intermediate_size, config.text_config.hidden_size, bias=False)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)

    def forward(self, audio_features):
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@auto_docstring(
    custom_intro="""
    The Voxtral model, which consists of Whisper encoder, a multi-modal projector and a LLama language model.
    """
)
class DixtralForConditionalGeneration(DixtralPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _keep_in_fp32_modules_strict = ["embed_positions"]

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.audio_tower = DixtralEncoder(config.audio_config)
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.multi_modal_projector = VoxtralMultiModalProjector(config)

        self._init_dicow_components(config)
        # Initialize weights and apply final processing
        self.post_init()

    def _init_dicow_components(self, config):
        self.ctc_weight = config.audio_config.ctc_weight

        # Additional layers for CTC
        if config.audio_config.additional_layer and self.ctc_weight > 0.0:
            custom_conf = copy.deepcopy(config.audio_config)
            custom_conf.d_model = config.text_config.hidden_size
            custom_conf.encoder_attention_heads = config.text_config.num_attention_heads
            custom_conf.encoder_ffn_dim = custom_conf.d_model * 2
            self.additional_layer = VoxtralEncoderLayer(custom_conf)

        if config.audio_config.additional_self_attention_layer and self.ctc_weight > 0.0:
            self.additional_self_attention_layer = VoxtralAttention(
                embed_dim=config.text_config.hidden_size,
                num_heads=config.text_config.num_attention_heads,
                dropout=config.text_config.attention_dropout,
                config=config.audio_config,  # Fixed: pass audio_config which is VoxtralConfig
            )

        # CTC head
        if self.ctc_weight > 0.0:
            self.ctc_lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size + 1, bias=False)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_audio_embeds(self, input_features: torch.FloatTensor, stno_mask: torch.FloatTensor):
        """
        This method is used to get the audio embeddings from input features (a log mel spectrogram), meaning inferring the audio encoder and the multi-modal projector.
        Args:
            input_features (`torch.FloatTensor`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]

        Returns:
            `torch.FloatTensor`:
                The audio embeddings.
        """
        audio_outputs = self.audio_tower(input_features, stno_mask=stno_mask)
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_hidden_states = audio_hidden_states.reshape(-1, self.config.audio_config.intermediate_size)
        audio_embeds = self.multi_modal_projector(audio_hidden_states)
        return audio_embeds

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer


    def possibly_update_last_hidden_states(self, hidden_states):
        """DiCoW post-processing for CTC"""
        if not self.config.audio_config.use_dicow_encoder:
            return hidden_states

        if hasattr(self, "additional_layer"):
            hidden_states, _ = self.additional_layer(
                hidden_states,
                attention_mask=None,
                layer_head_mask=None,
                output_attentions=False,
            )
        elif hasattr(self, "additional_self_attention_layer"):
            hidden_states, _ = self.additional_self_attention_layer(
                hidden_states,
                attention_mask=None,
                layer_head_mask=None,
                output_attentions=False,
            )

        return hidden_states

    def get_enc_logits(self, hidden_states):
        """
        Get CTC logits from encoder hidden states.
        Applies optional additional processing layer and projects to vocabulary.

        Args:
            hidden_states: Encoder output hidden states

        Returns:
            logits: CTC logits of shape (batch_size, seq_len, vocab_size + 1)
        """
        hidden_states = self.possibly_update_last_hidden_states(hidden_states)
        logits = self.ctc_lm_head(hidden_states)
        return logits

    def right_pad_labels(self, labels, pad_value=-100):
        """
        labels: (B, L) tensor possibly left/right padded
        returns: right-padded labels only
        """
        B, L = labels.shape
        new_labels = torch.full_like(labels, pad_value)
        max_len = 1
        for b in range(B):
            valid = labels[b][labels[b] != pad_value]
            max_len = max(max_len, len(valid))
            new_labels[b, :valid.numel()] = valid

        new_labels = new_labels[:, :max_len]

        return new_labels

    def get_ctc_loss(self, logits, labels, input_lengths):

        """Compute CTC loss for DiCoW"""
        if labels.max() >= self.config.text_config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.text_config.vocab_size}")

        # Assuming that padded tokens are filled with -100
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)

        # CTC loss doesn't support fp16
        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

        with torch.backends.cudnn.flags(enabled=True):
            ctc_loss = nn.functional.ctc_loss(
                log_probs,
                labels,
                input_lengths,
                target_lengths,
                blank=logits.shape[-1] - 1,
                reduction=self.config.audio_config.ctc_loss_reduction,
                zero_infinity=True,
            )

        return ctc_loss

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        stno_mask=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import VoxtralForConditionalGeneration, AutoProcessor
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> repo_id = "mistralai/Voxtral-Mini-3B-2507"

        >>> processor = AutoProcessor.from_pretrained(repo_id)
        >>> model = VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)

        >>> conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav",
                    },
                    {"type": "text", "text": "What can you tell me about this audio?"},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(conversation)
        >>> inputs = inputs.to(device, dtype=torch.bfloat16)

        >>> outputs = model.generate(**inputs, max_new_tokens=30)
        >>> processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        ["This audio is a humorous conversation between two friends, likely in English, where one of them is trying to figure out what the other's tattoo says."]
        ```"""
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        ctc_loss = None
        if input_features is not None:
            # Get audio encoder outputs
            audio_outputs = self.audio_tower(input_features, stno_mask=stno_mask)
            audio_hidden_states = audio_outputs.last_hidden_state

            # Project audio features for language model
            audio_hidden_states_flat = audio_hidden_states.reshape(-1, self.config.audio_config.intermediate_size)
            audio_embeds_flat = self.multi_modal_projector(audio_hidden_states_flat)

            # Replace text-audio token placeholders with audio embeddings
            audio_token_mask = input_ids == self.config.audio_token_id
            inputs_embeds[audio_token_mask] = audio_embeds_flat

            # Compute CTC loss on projected embeddings if configured
            if (self.config.audio_config.use_dicow_encoder and
                    self.config.audio_config.ctc_weight > 0.0 and
                    labels is not None and
                    self.training and
                    audio_token_mask is not None):

                # Create tensor with shape of input_ids filled with zeros
                batch_size, seq_len = input_ids.shape
                hidden_dim = audio_embeds_flat.shape[-1]
                ctc_embeds = torch.empty(
                    batch_size, seq_len, hidden_dim,
                    device=audio_embeds_flat.device,
                    dtype=audio_embeds_flat.dtype
                )

                # Fill with audio_embeds at audio_token positions
                ctc_embeds[audio_token_mask] = audio_embeds_flat

                # Remove values outside maximum valid range using audio_mask
                enc_output_lens = audio_token_mask.sum(dim=1)
                max_valid_len = enc_output_lens.max().item()
                first_audio_token = audio_token_mask.int().argmax(dim=1).min().item()  # First True position per batch
                ctc_embeds = ctc_embeds[:, first_audio_token:first_audio_token+max_valid_len, :]

                # Get encoder logits for CTC
                enc_lm_logits = self.get_enc_logits(ctc_embeds)

                # Prepare encoder labels
                enc_labels = labels.clone()

                # Replace EOS tokens with ignore index
                enc_labels[enc_labels == self.config.text_config.eos_token_id] = -100
                enc_labels = self.right_pad_labels(enc_labels)

                # Compute CTC loss
                ctc_loss = self.get_ctc_loss(enc_lm_logits, enc_labels, enc_output_lens)

        outputs: BaseModelOutputWithPast = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if ctc_loss is not None and outputs.loss is not None:
            if wandb.run is not None:
                wandb.log({"dec_loss": outputs.loss, "ctc_loss": ctc_loss})
            total_loss = outputs.loss + self.config.audio_config.ctc_weight * ctc_loss
            outputs.loss = total_loss
        elif ctc_loss is not None:
            outputs.loss = self.config.audio_config.ctc_weight * ctc_loss


        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Overwritten -- we should not pass input_features when we are in cached decoding stage

        input_features = kwargs.pop("input_features", None)
        cache_position = kwargs.get("cache_position")

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if cache_position is not None and cache_position[0] == 0:
            # input_features should only be passed when we are not in cached decoding stage
            model_inputs["input_features"] = input_features

        return model_inputs


__all__ = ["DixtralPreTrainedModel", "DixtralEncoder", "DixtralForConditionalGeneration"]