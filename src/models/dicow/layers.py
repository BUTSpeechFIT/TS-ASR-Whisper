import torch
from torch import nn
import math
from transformers.models.whisper.modeling_whisper import WhisperAttention
from transformers.activations import ACT2FN

class CustomLinear(nn.Linear):
    def __init__(self, *args, init_eye_val=0.0, fddt_init=None, init_fun=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_eye_val = init_eye_val
        self.fddt_init = fddt_init
        self.init_fun = init_fun

    def reset_parameters(self) -> None:
        if self.init_fun is not None:
            self.init_fun()
        else:
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.bias)

            if self.fddt_init == 'non-disturbing':
                self.weight.data = torch.eye(*self.weight.shape).data
                if self.bias is not None:
                    self.bias.data.zero_()
            elif self.fddt_init == 'suppressive':
                eye = torch.eye(*self.weight.shape)
                eye *= self.init_eye_val
                self.weight.data = eye.data
                if self.bias is not None:
                    self.bias.data.zero_()

class CustomDiagonalLinear(nn.Module):
    def __init__(self, d_model, bias=True, init_eye_val=0.0, fddt_init=None):
        super().__init__()
        self.init_eye_val = init_eye_val
        self.weight = nn.Parameter(torch.full((d_model,), init_eye_val))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.fddt_init = fddt_init
        self.reset_parameters()

    def reset_parameters(self):
        fan = self.weight.size(0)
        bound = math.sqrt(3.0 / fan)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.zeros_(self.bias)
        if self.fddt_init == 'non-disturbing':
            self.weight.data = torch.ones_like(self.weight.data).data
            if self.bias is not None:
                self.bias.data.zero_()
        elif self.fddt_init == 'suppressive':
            self.weight.data = self.init_eye_val * torch.ones_like(self.weight.data).data
            if self.bias is not None:
                self.bias.data.zero_()

    def forward(self, input):
        out = input * self.weight
        if self.bias is not None:
            out += self.bias
        return out


class Gate(nn.Module):
    def __init__(self, items, init_val=0.0):
        super().__init__()
        self.init_val = init_val
        self.gate = nn.Parameter(torch.full((items,), init_val))
        self.reset_parameters()

    def forward(self, input, dim):
        if input.ndim != 4:
            raise ValueError('input must be a 4D tensor')
        shape = [1] * 4
        shape[dim] = -1
        return input * self.gate.view(*shape)

    def reset_parameters(self):
        self.gate.data = self.init_val * torch.ones_like(self.gate.data).data



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

        self.cross_attn = WhisperAttention(
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
            CustomLinear(self.embed_dim * 2, self.ffn_dim, init_fun=first_init_fun),
            ACT2FN[config.activation_function],
            nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1),
            CustomLinear(self.ffn_dim, self.embed_dim, init_fun=second_init_fun),
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
