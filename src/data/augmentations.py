import math
import os
import pathlib
import random
from typing import Optional, Sequence, Union

import torch
import torchaudio


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def mask_along_axis(
        spec: torch.Tensor,
        spec_lengths: torch.Tensor,
        mask_width_range: Sequence[int] = (0, 30),
        dim: int = 1,
        num_mask: int = 2,
        replace_with_zero: bool = True,
):
    """Apply mask along the specified direction.

    Args:
        spec: (Batch, Length, Freq)
        spec_lengths: (Length): Not using lengths in this implementation
        mask_width_range: Select the width randomly between this range
    """

    org_size = spec.size()
    if spec.dim() == 4:
        # spec: (Batch, Channel, Length, Freq) -> (Batch * Channel, Length, Freq)
        spec = spec.view(-1, spec.size(2), spec.size(3))

    B = spec.shape[0]
    # D = Length or Freq
    D = spec.shape[dim]
    # mask_length: (B, num_mask, 1)
    mask_length = torch.randint(
        mask_width_range[0],
        mask_width_range[1],
        (B, num_mask),
        device=spec.device,
    ).unsqueeze(2)

    # mask_pos: (B, num_mask, 1)
    mask_pos = torch.randint(0, max(1, D - mask_length.max()), (B, num_mask), device=spec.device).unsqueeze(2)

    # aran: (1, 1, D)
    aran = torch.arange(D, device=spec.device)[None, None, :]
    # mask: (Batch, num_mask, D)
    mask = (mask_pos <= aran) * (aran < (mask_pos + mask_length))
    # Multiply masks: (Batch, num_mask, D) -> (Batch, D)
    mask = mask.any(dim=1)
    if dim == 1:
        # mask: (Batch, Length, 1)
        mask = mask.unsqueeze(2)
    elif dim == 2:
        # mask: (Batch, 1, Freq)
        mask = mask.unsqueeze(1)

    if replace_with_zero:
        value = 0.0
    else:
        value = spec.mean()

    if spec.requires_grad:
        spec = spec.masked_fill(mask, value)
    else:
        spec = spec.masked_fill_(mask, value)
    spec = spec.view(*org_size)
    return spec, spec_lengths


DEFAULT_TIME_WARP_MODE = "bicubic"


def time_warp(x: torch.Tensor, window: int = 80, mode: str = DEFAULT_TIME_WARP_MODE):
    """Time warping using torch.interpolate.

    Args:
        x: (Batch, Time, Freq)
        window: time warp parameter
        mode: Interpolate mode
    """

    # bicubic supports 4D or more dimension tensor
    org_size = x.size()
    if x.dim() == 3:
        # x: (Batch, Time, Freq) -> (Batch, 1, Time, Freq)
        x = x[:, None]

    t = x.shape[2]
    if t - window <= window:
        return x.view(*org_size)

    center = torch.randint(window, t - window, (1,))[0]
    warped = torch.randint(center - window, center + window, (1,))[0] + 1

    # left: (Batch, Channel, warped, Freq)
    # right: (Batch, Channel, time - warped, Freq)
    left = torch.nn.functional.interpolate(x[:, :, :center], (warped, x.shape[3]), mode=mode, align_corners=False)
    right = torch.nn.functional.interpolate(x[:, :, center:], (t - warped, x.shape[3]), mode=mode, align_corners=False)

    if x.requires_grad:
        x = torch.cat([left, right], dim=-2)
    else:
        x[:, :, :warped] = left
        x[:, :, warped:] = right

    return x.view(*org_size)


class TimeWarp(torch.nn.Module):
    """Time warping using torch.interpolate.

    Args:
        window: time warp parameter
        mode: Interpolate mode
    """

    def __init__(self, window: int = 80, mode: str = DEFAULT_TIME_WARP_MODE):
        super().__init__()
        self.window = window
        self.mode = mode

    def extra_repr(self):
        return f"window={self.window}, mode={self.mode}"

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor = None):
        """Forward function.

        Args:
            x: (Batch, Time, Freq)
            x_lengths: (Batch,)
        """

        if x_lengths is None or all(le == x_lengths[0] for le in x_lengths):
            # Note that applying same warping for each sample
            y = time_warp(x, window=self.window, mode=self.mode)
        else:
            # FIXME(kamo): I have no idea to batchify Timewarp
            ys = []
            for i in range(x.size(0)):
                _y = time_warp(
                    x[i][None, : x_lengths[i]],
                    window=self.window,
                    mode=self.mode,
                )[0]
                ys.append(_y)
            y = pad_list(ys, 0.0)

        return y, x_lengths


class MaskAlongAxis(torch.nn.Module):
    def __init__(
            self,
            mask_width_range: Union[int, Sequence[int]] = (0, 30),
            num_mask: int = 2,
            dim: Union[int, str] = "time",
            replace_with_zero: bool = True,
    ):
        if isinstance(mask_width_range, int):
            mask_width_range = (0, mask_width_range)
        if len(mask_width_range) != 2:
            raise TypeError(
                f"mask_width_range must be a tuple of int and int values: " f"{mask_width_range}",
            )

        if mask_width_range[1] < mask_width_range[0]:
            raise ValueError("mask_width_range must be (min_width, max_width)")
        if isinstance(dim, str):
            if dim == "time":
                dim = 1
            elif dim == "freq":
                dim = 2
            else:
                raise ValueError("dim must be int, 'time' or 'freq'")
        if dim == 1:
            self.mask_axis = "time"
        elif dim == 2:
            self.mask_axis = "freq"
        else:
            self.mask_axis = "unknown"

        super().__init__()
        self.mask_width_range = mask_width_range
        self.num_mask = num_mask
        self.dim = dim
        self.replace_with_zero = replace_with_zero

    def extra_repr(self):
        return f"mask_width_range={self.mask_width_range}, " f"num_mask={self.num_mask}, axis={self.mask_axis}"

    def forward(self, spec: torch.Tensor, spec_lengths: torch.Tensor = None):
        """Forward function.

        Args:
            spec: (Batch, Length, Freq)
        """

        return mask_along_axis(
            spec,
            spec_lengths,
            mask_width_range=self.mask_width_range,
            dim=self.dim,
            num_mask=self.num_mask,
            replace_with_zero=self.replace_with_zero,
        )


class MaskAlongAxisVariableMaxWidth(torch.nn.Module):
    """Mask input spec along a specified axis with variable maximum width.

    Formula:
        max_width = max_width_ratio * seq_len
    """

    def __init__(
            self,
            mask_width_ratio_range: Union[float, Sequence[float]] = (0.0, 0.05),
            num_mask: int = 2,
            dim: Union[int, str] = "time",
            replace_with_zero: bool = True,
    ):
        if isinstance(mask_width_ratio_range, float):
            mask_width_ratio_range = (0.0, mask_width_ratio_range)
        if len(mask_width_ratio_range) != 2:
            raise TypeError(
                f"mask_width_ratio_range must be a tuple of float and float values: " f"{mask_width_ratio_range}",
            )

        if mask_width_ratio_range[1] < mask_width_ratio_range[0]:
            raise ValueError("mask_width_ratio_range must be (min_ratio, max_ratio)")
        if isinstance(dim, str):
            if dim == "time":
                dim = 1
            elif dim == "freq":
                dim = 2
            else:
                raise ValueError("dim must be int, 'time' or 'freq'")
        if dim == 1:
            self.mask_axis = "time"
        elif dim == 2:
            self.mask_axis = "freq"
        else:
            self.mask_axis = "unknown"

        super().__init__()
        self.mask_width_ratio_range = mask_width_ratio_range
        self.num_mask = num_mask
        self.dim = dim
        self.replace_with_zero = replace_with_zero

    def extra_repr(self):
        return (
            f"mask_width_ratio_range={self.mask_width_ratio_range}, " f"num_mask={self.num_mask}, axis={self.mask_axis}"
        )

    def forward(self, spec: torch.Tensor, spec_lengths: torch.Tensor = None):
        """Forward function.

        Args:
            spec: (Batch, Length, Freq)
        """

        max_seq_len = spec.shape[self.dim]
        min_mask_width = math.floor(max_seq_len * self.mask_width_ratio_range[0])
        min_mask_width = max([0, min_mask_width])
        max_mask_width = math.floor(max_seq_len * self.mask_width_ratio_range[1])
        max_mask_width = min([max_seq_len, max_mask_width])

        if max_mask_width > min_mask_width:
            return mask_along_axis(
                spec,
                spec_lengths,
                mask_width_range=(min_mask_width, max_mask_width),
                dim=self.dim,
                num_mask=self.num_mask,
                replace_with_zero=self.replace_with_zero,
            )
        return spec, spec_lengths


class SpecAug(torch.nn.Module):
    """Implementation of SpecAug.

    Reference:
        Daniel S. Park et al.
        "SpecAugment: A Simple Data
         Augmentation Method for Automatic Speech Recognition"

    .. warning::
        When using cuda mode, time_warp doesn't have reproducibility
        due to `torch.nn.functional.interpolate`.

    """

    def __init__(
            self,
            apply_time_warp: bool = True,
            time_warp_window: int = 5,
            time_warp_mode: str = "bicubic",
            apply_freq_mask: bool = True,
            freq_mask_width_range: Union[int, Sequence[int]] = (0, 20),
            num_freq_mask: int = 2,
            apply_time_mask: bool = True,
            time_mask_width_range: Optional[Union[int, Sequence[int]]] = None,
            time_mask_width_ratio_range: Optional[Union[float, Sequence[float]]] = None,
            num_time_mask: int = 2,
    ):
        if not apply_time_warp and not apply_time_mask and not apply_freq_mask:
            raise ValueError("Either one of time_warp, time_mask, or freq_mask should be applied")
        if apply_time_mask and (time_mask_width_range is not None) and (time_mask_width_ratio_range is not None):
            raise ValueError('Either one of "time_mask_width_range" or ' '"time_mask_width_ratio_range" can be used')
        super().__init__()
        self.apply_time_warp = apply_time_warp
        self.apply_freq_mask = apply_freq_mask
        self.apply_time_mask = apply_time_mask

        if apply_time_warp:
            self.time_warp = TimeWarp(window=time_warp_window, mode=time_warp_mode)
        else:
            self.time_warp = None

        if apply_freq_mask:
            self.freq_mask = MaskAlongAxis(
                dim="freq",
                mask_width_range=freq_mask_width_range,
                num_mask=num_freq_mask,
            )
        else:
            self.freq_mask = None

        if apply_time_mask:
            if time_mask_width_range is not None:
                self.time_mask = MaskAlongAxis(
                    dim="time",
                    mask_width_range=time_mask_width_range,
                    num_mask=num_time_mask,
                )
            elif time_mask_width_ratio_range is not None:
                self.time_mask = MaskAlongAxisVariableMaxWidth(
                    dim="time",
                    mask_width_ratio_range=time_mask_width_ratio_range,
                    num_mask=num_time_mask,
                )
            else:
                raise ValueError(
                    'Either one of "time_mask_width_range" or ' '"time_mask_width_ratio_range" should be used.'
                )
        else:
            self.time_mask = None

    def forward(self, x, x_lengths=None):
        if x.ndim == 2:
            x = x[None, ...]
        elif x.ndim != 3:
            raise ValueError("Please ensure input has (T,H) or (B,T,H) shape.")
        if self.time_warp is not None:
            x, x_lengths = self.time_warp(x, x_lengths)
        if self.freq_mask is not None:
            out, x_lengths = self.freq_mask(x[:, :, :80], x_lengths)
            x[:, :, :80] = out
            # x, x_lengths = self.freq_mask(x, x_lengths)
        if self.time_mask is not None:
            out, x_lengths = self.time_mask(x[:, :, :80], x_lengths)
            x[:, :, :80] = out
        return x, x_lengths


class RandomBackgroundNoise:
    def __init__(self, sample_rate, noise_dir, min_snr_db=0, max_snr_db=15):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        if not os.path.exists(noise_dir):
            raise IOError(f'Noise directory `{noise_dir}` does not exist')
        # find all WAV files including in sub-folders:
        self.noise_files_list = list(pathlib.Path(noise_dir).glob('**/*.wav'))
        if len(self.noise_files_list) == 0:
            raise IOError(f'No .wav file found in the noise directory `{noise_dir}`')

    def __call__(self, audio_data, vad_mask):
        random_noise_file = random.choice(self.noise_files_list)

        effects = [
            ['remix', '1'],  # convert to mono
            ['rate', str(self.sample_rate)],  # resample
        ]
        noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)
        audio_length = audio_data.shape[-1]
        noise_length = noise.shape[-1]
        if noise_length > audio_length:
            offset = random.randint(0, noise_length - audio_length)
            noise = noise[..., offset:offset + audio_length]
        elif noise_length < audio_length:
            noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length - noise_length))], dim=-1)

        # if "speech" in str(random_noise_file):
        #     x=4
        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = audio_data.text_norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / audio_power

        return (scale * audio_data + noise) / 2, vad_mask


class RandomSpeedChange:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, audio_data):
        speed_factor = random.choice([0.9, 1.0, 1.1])
        if speed_factor == 1.0:  # no change
            return audio_data

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_data, self.sample_rate, sox_effects)
        return transformed_audio
