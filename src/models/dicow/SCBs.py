import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.whisper.modeling_whisper import WHISPER_ATTENTION_CLASSES

from .layers import CustomLinear, CustomDiagonalLinear, CustomLinearInitialized

def first_init_fun(module):
    # Zero out all weights initially
    # module.weight.data.zero_()
    torch.nn.init.xavier_uniform_(module.weight, gain=0.1)

    # Create identity mapping for second half of input (q_normed part)
    # Input: [cross_attn_output, q_normed] -> map q_normed to first embed_dim outputs
    module.weight.data[:module.weight.shape[1] // 2, module.weight.shape[1] // 2:] += torch.eye(
        module.weight.shape[1] // 2)

    # Zero bias
    module.bias.data.zero_()


def second_init_fun(module):
    # module.weight.data.zero_()
    torch.nn.init.xavier_uniform_(module.weight, gain=0.1)

    # Create identity mapping from first embed_dim inputs to output
    module.weight.data[:, :module.weight.shape[0]] += torch.eye(module.weight.shape[0])

    # Zero bias for second linear
    module.bias.data.zero_()


# Cross attention block that can easily learn to ignore cross attention initially
class CrossAttentionEnrollBlockNew(nn.Module):
    def __init__(self, config, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.embed_dim = config.d_model
        self.ffn_dim = config.encoder_ffn_dim

        self.cross_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )

        # Layer normalization (pre-norm style)
        # self.norm_attn = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)
        self.cross_gate = nn.Parameter(torch.zeros(1))
        # Feed-forward network that maps concat space back to single channel
        self.ffn = nn.Sequential(
            CustomLinearInitialized(self.embed_dim * 2, self.ffn_dim, init_fun=first_init_fun),
            ACT2FN[config.activation_function],
            nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1),
            CustomLinearInitialized(self.ffn_dim, self.embed_dim, init_fun=second_init_fun),
            nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, 2, T, F) - batch, channels, time, features
        Returns:
            Updated hidden states of same shape
        """
        q_channel = hidden_states[:, 0]  # (B, T, F)
        kv_channel = hidden_states[:, 1]  # (B, T, F)

        # Cross-attention
        attn_output = self.cross_attn(
            hidden_states=q_channel,
            key_value_states=kv_channel,
            output_attentions=False
        )[0]

        # Concatenate attention output with original normalized query
        q_concat = torch.cat([attn_output, q_channel], dim=-1)  # (B, T, 2*F)

        # Feed-forward processing (no normalization to preserve initialization)
        # updated_q = self.ffn(q_concat)  # (B, T, F)
        updated_q = q_channel + torch.tanh(self.cross_gate) * self.ffn(q_concat)

        # Return stacked result (only query channel is updated)
        return torch.stack([updated_q, kv_channel], dim=1)


class SpeakerCommunicationBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mt_num_speakers = getattr(config, "mt_num_speakers", 2)
        self.embed_dim = config.d_model
        self.scb_method = config.scb_method
        self.config = config

        if self.scb_method == "cross_attention_enroll_new":
            self.method = CrossAttentionEnrollBlockNew(config)
        elif self.scb_method == "identity":
            self.method = (nn.Parameter(torch.zeros(self.embed_dim)) if config.fddt_bias_only else (
                CustomDiagonalLinear(self.embed_dim, bias=True,
                                     init_eye_val=1.0) if config.fddt_is_diagonal else CustomLinear(
                    self.embed_dim, self.embed_dim, bias=True, init_eye_val=1.0)))
        else:
            raise ValueError(f"Unsupported scb_method: {self.scb_method}")

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape
        S = self.mt_num_speakers

        # Reshape to (B//S, S, T, F)
        x_reshaped = x.view(B // S, S, T, F)

        # Call the selected method
        out = self.method(x_reshaped)

        # Reshape back (B, T, F)
        out_merged = out.view(B, T, F)
        return out_merged
