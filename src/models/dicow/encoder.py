import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperEncoderLayer, WhisperAttention
from .FDDT import FDDT
from .config import DiCoWConfig
from .layers import CustomLinear, CustomDiagonalLinear, Gate


class DiCoWEncoder(WhisperEncoder):
    config_class = DiCoWConfig

    def __init__(self, config: DiCoWConfig):
        super().__init__(config)
        self.ctc_weight = config.ctc_weight
        if config.additional_layer and self.ctc_weight > 0.0:
            self.additional_layer = WhisperEncoderLayer(config)
        if config.additional_self_attention_layer and self.ctc_weight > 0.0:
            self.additional_self_attention_layer = WhisperAttention(
                embed_dim=config.d_model,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config,
            )
        if config.sub_sample and self.ctc_weight > 0.0:
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
        if self.ctc_weight > 0.0:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size + 1, bias=False)
        self.final_dropout = nn.Dropout(config.final_dropout)
        if config.use_fddt:
            num_fddts = self.config.apply_fddt_to_n_layers if self.config.apply_fddt_to_n_layers != -1 else len(
                self.layers)
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
        self.first_task_token = self.config.vocab_size - 30 * 50 - 1 - 6  # 30 seconds of 50 Hz timestamps -1 to get to 0.0 and -6 number of tasks
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, CustomLinear) or isinstance(module, CustomDiagonalLinear) or isinstance(module, Gate):
            module.reset_parameters()

    def get_output_embeddings(self):
        return None

    def possibly_update_last_hidden_states(self, hidden_states):
        if hasattr(self, "additional_layer"):
            hidden_states, = self.additional_layer(
                hidden_states,
                attention_mask=None,
                output_attentions=False,
                layer_head_mask=None,
            )
        elif hasattr(self, "additional_self_attention_layer"):
            hidden_states, _ = self.additional_self_attention_layer(
                hidden_states,
                attention_mask=None,
                output_attentions=False,
                layer_head_mask=None,
            )

        hidden_states = self.final_dropout(hidden_states)
        if hasattr(self, "subsample_conv2"):
            hidden_states = self.subsample_conv2(self.subsample_conv1(hidden_states.transpose(1, 2))).transpose(1, 2)
        return hidden_states

    def get_loss(self, logits, labels):
        if labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")
        if self.config.remove_timestamps_from_ctc:
            labels = torch.nn.utils.rnn.pad_sequence([label[label < self.first_task_token] for label in labels],
                                                     padding_value=-100).T
        input_lengths = torch.full((logits.shape[0],), fill_value=logits.shape[1],
                                   device=logits.device)

        # assuming that padded tokens are filled with -100
        # when not being attended to
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)

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
            stno_mask=None,
    ):
        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
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

        """<DiCoW CODE>"""
        if self.config.use_fddt:
            inputs_embeds = self.initial_fddt(inputs_embeds, stno_mask)
        """</DiCoW CODE>"""

        all_positions = torch.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)

        hidden_states = inputs_embeds + self.embed_positions(all_positions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), (
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                """<DiCoW CODE>"""
                if self.config.use_fddt and idx < len(self.fddts):
                    hidden_states = self.fddts[idx](hidden_states, stno_mask)
                """</DiCoW CODE>"""

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
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
