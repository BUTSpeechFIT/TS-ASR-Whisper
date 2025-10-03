import torch
from torch import nn


class CustomLinear(nn.Linear):
    def __init__(self, *args, init_eye_val=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_eye_val = init_eye_val


class CustomLinearInitialized(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, init_fun=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.init_fun = init_fun


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


class Gate(nn.Module):
    def __init__(self, items, init_val=0.0):
        super().__init__()
        self.init_val = init_val
        self.gate = nn.Parameter(torch.full((items,), init_val))

    def forward(self, input, dim):
        if input.ndim != 4:
            raise ValueError('input must be a 4D tensor')
        shape = [1] * 4
        shape[dim] = -1
        return input * self.gate.view(*shape)
