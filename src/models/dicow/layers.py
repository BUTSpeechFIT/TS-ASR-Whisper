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
        self.reset_parameters()  # Ensure consistent init on creation

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # Apply custom init function if provided
            if hasattr(self,"init_fun") and self.init_fun is not None:
                self.init_fun(self)
                return

            # Default initialization
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

            if hasattr(self, "fddt_init"):
                # FDDT-specific inits
                if self.fddt_init == 'non-disturbing':
                    # Make weight an identity matrix (if possible)
                    if self.weight.shape[0] == self.weight.shape[1]:
                        self.weight.copy_(torch.eye(self.weight.shape[0], device=self.weight.device))
                    else:
                        # Not square — fill first min(n, m) diagonals
                        eye = torch.zeros_like(self.weight)
                        n = min(self.weight.shape)
                        eye[:n, :n] = torch.eye(n, device=self.weight.device)
                        self.weight.copy_(eye)

                elif self.fddt_init == 'suppressive':
                    if self.weight.shape[0] == self.weight.shape[1]:
                        self.weight.copy_(self.init_eye_val * torch.eye(self.weight.shape[0], device=self.weight.device))
                    else:
                        eye = torch.zeros_like(self.weight)
                        n = min(self.weight.shape)
                        eye[:n, :n] = self.init_eye_val * torch.eye(n, device=self.weight.device)
                        self.weight.copy_(eye)

class CustomDiagonalLinear(nn.Module):
    def __init__(self, d_model, bias=True, init_eye_val=0.0, fddt_init=None):
        super().__init__()
        self.init_eye_val = init_eye_val
        self.weight = nn.Parameter(torch.full((d_model,), init_eye_val))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.fddt_init = fddt_init
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            # random init
            fan = self.weight.size(0)
            bound = math.sqrt(3.0 / fan)
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.zero_()

            # custom modes
            if self.fddt_init == 'non-disturbing':
                self.weight.fill_(1.0)
            elif self.fddt_init == 'suppressive':
                self.weight.fill_(self.init_eye_val)

    def forward(self, input):
        out = input * self.weight
        if self.bias is not None:
            out += self.bias
        return out

class InterpolationGate(nn.Module):
    def __init__(self, items, init_val=0.0):
        super().__init__()
        self.init_val = init_val
        self.gate = nn.Parameter(torch.full((items,), init_val))
        self.reset_parameters()

    def forward(self, orig_seq, new_seq):
        gate_act = torch.nn.functional.sigmoid(self.gate)
        output = (1 - gate_act) * orig_seq + gate_act * new_seq
        return output

    def reset_parameters(self):
        with torch.no_grad():
            self.gate.fill_(self.init_val)

def propagate_first_half_embeds_init(module):
    # Zero out all weights initially
    # module.weight.data.zero_()
    torch.nn.init.xavier_uniform_(module.weight, gain=1e-3)

    # Create identity mapping for first half of input (q_orig)
    # Input: [q_orig, cross_attn_output] -> map q_orig to first embed_dim outputs
    module.weight.data[:module.weight.shape[1] // 2, :module.weight.shape[1] // 2] += torch.eye(
        module.weight.shape[1] // 2)

    # Zero bias
    module.bias.data.zero_()


def propage_first_embeds_to_match_output_dim_init(module):
    # module.weight.data.zero_()
    torch.nn.init.xavier_uniform_(module.weight, gain=1e-3)

    # Create identity mapping from first embed_dim inputs to output
    module.weight.data[:, :module.weight.shape[0]] += torch.eye(module.weight.shape[0])

    # Zero bias for second linear
    module.bias.data.zero_()


# Cross attention block that can easily learn to ignore cross attention initially
class CrossAttentionEnrollBlock(nn.Module):
    def __init__(self, config):
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
        self.cross_gate = InterpolationGate(1,init_val=-1.0)
        # Feed-forward network that maps concat space back to single channel
        self.ffn = nn.Sequential(
            CustomLinear(self.embed_dim * 2, self.ffn_dim, init_fun=propagate_first_half_embeds_init),
            ACT2FN[config.activation_function],
            nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1),
            CustomLinear(self.ffn_dim, self.embed_dim, init_fun=propage_first_embeds_to_match_output_dim_init),
            nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, 2, T, F) - batch, channels, time, features
        Returns:
            Updated hidden states of same shape
        """
        q = hidden_states[:, 0]  # (B, T, F)
        kv = hidden_states[:, 1]  # (B, T, F)

        # Cross-attention
        attn_output = self.cross_attn(
            hidden_states=q,
            key_value_states=kv,
            output_attentions=False
        )[0]

        # Concatenate attention output with original normalized query
        q_concat = torch.cat([q, attn_output], dim=-1)  # (B, T, 2*F)

        # Feed-forward processing (no normalization to preserve initialization)
        updated_q = self.ffn(q_concat)  # (B, T, F)

        q_out = self.cross_gate(q, updated_q)
        # Return stacked result (only query channel is updated)
        return torch.stack([q_out, kv], dim=1)

class SpeakerCommunicationBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_speakers = getattr(config, "mt_num_speakers", 2)
        self.config = config

        self.cae = CrossAttentionEnrollBlock(config)

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape
        S = self.num_speakers

        # Reshape to (B//S, S, T, F)
        x_reshaped = x.view(B//S, S, T, F)

        # Call the selected method
        out = self.cae(x_reshaped)

        # Reshape back (B, T, F)
        out_merged = out.view(B, T, F)
        return out_merged


if __name__ == "__main__":
    model1 = CustomLinear(16 * 2, 64, init_fun=propagate_first_half_embeds_init)
    model2 = CustomLinear(64, 16, init_fun=propage_first_embeds_to_match_output_dim_init)
    input1 = torch.ones(16, 16)
    input2 = torch.zeros(16, 16)
    input = torch.concat((input1, input2), dim=-1)
    output = model2(model1(input))
    print(f"Mean err: {(input1-output).mean()}")


    model_1 = CustomDiagonalLinear(4, bias=False, fddt_init='suppressive', init_eye_val=0.1)
    model_2 = CustomDiagonalLinear(4, bias=False, fddt_init='suppressive', init_eye_val=0.1)
    model_3 = CustomDiagonalLinear(4, bias=False, fddt_init='suppressive', init_eye_val=0.1)
    model_4 = CustomDiagonalLinear(4, bias=False, fddt_init='suppressive', init_eye_val=0.1)
    model = nn.Sequential(model_1, model_2, model_3, model_4)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model_1.reset_parameters()


    x = torch.ones(2, 4)
    y = torch.ones(2, 4)

    for i in range(100):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        print(f"Step {i}: mean weight {model_1.weight.mean().item():.4f}")