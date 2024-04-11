from functools import lru_cache
from typing import Tuple, List, Optional, Union

import torch
import numpy as np
import scipy

import torch.nn.functional as F


class Compose:
    def __init__(self, transforms, prob=1.):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, img):
        if type(self.prob) == type(0.5): 
            for t in self.transforms:
                if torch.rand(1) < self.prob:
                    img = t(img)
        elif type(self.prob) == type([0.5, 0.5]):
            assert len(self.transforms) == len(self.prob)
            for i, t in enumerate(self.transforms):
                if torch.rand(1) < self.prob[i]:
                    img = t(img)
        return img

class InstanceNorm:
    def __init__(
        self,
        mean: bool = True,
        std: bool = True,
        dim: int = -1,
        eps: float = 1e-12,
    ) -> None:
        """
        Apply instance normalization.

        Args:
            mean: subtract by mean
            std: scale by standard deviation
            dim: dimension to compute stats across; default to dim=0 for computing across samples
            eps: small number for division
            apply_to: streams to apply transform to
        """
        super().__init__()
        self.mean = mean
        self.dim = dim
        self.std = std
        self.eps = eps

    def __call__(self, data):
        if self.mean:
            data = data - torch.mean(data, axis=self.dim, keepdims=True)
        if self.std:
            data = data / (torch.std(data, axis=self.dim, keepdims=True) + self.eps)
        return data

    def __repr__(self) -> str:
        return self.repr_helper(param=f"mean={self.mean}, std={self.std}, dim={self.dim}, eps={self.eps}")

class Norm:
    def __init__(
        self,
        mean = 0,
        std = 50,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, data):
        data = data - self.mean
        data = data / self.std
        return data

    def __repr__(self) -> str:
        return self.repr_helper(param=f"mean={self.mean}, std={self.std}, dim={self.dim}, eps={self.eps}")

class Jitter:
    def __init__(
        self,
        sigma: float = 0.8,
    ) -> None:
        """
        """
        super().__init__()
        self.sigma = sigma

    def __call__(self, data):
        jitter_noise = torch.normal(0, self.sigma, size=data.shape).to(data.device)
        return data + jitter_noise

    def __repr__(self) -> str:
        return self.repr_helper(param=f"sigma={self.sigma}")

class Scale:
    def __init__(
        self,
        sigma: float = 1.1,
    ) -> None:
        """
        """
        super().__init__()
        self.sigma = sigma

    def __call__(self, data):
        scale = torch.normal(1, self.sigma, size=data.shape).to(data.device)
        # scale_rp = repeat(scale, 'b s -> b c s', c=data.shape[1]).to(data.device)
        return data * scale

    def __repr__(self) -> str:
        return self.repr_helper(param=f"sigma={self.sigma}")

class ZeroMasking:
    def __init__(
        self,
        max_length=200,
        dim=-1,
        ) -> None:
        super().__init__()
        self.max_length = max_length
        self.dim = dim

    def __call__(self, data):
        mask_length = torch.randint(high=self.max_length, size=(1,))
        data_length = data.shape[self.dim]
        begin_point = torch.randint(low=0, high=int(data_length-mask_length), size=(1,))

        data[:, int(begin_point):int(begin_point+mask_length)] = 0.
        print(mask_length)
        return data

class Downsample:
    def __init__(
        self,
        new_size = None,
        new_scale = None,
    ) -> None:
        """
        """
        super().__init__()
        self.new_size = new_size
        self.new_scale = new_scale
        assert new_scale != new_size

    def __call__(self, data):
        new_data = F.interpolate(data[None], size=self.new_size, scale_factor=self.new_scale, mode='linear')
        return new_data[0]

@lru_cache(maxsize=None)
def design_filter(
    cutoff_hz: Union[float, Tuple[float]],
    sampling_rate_hz: Optional[float] = 100,
    num_taps: Optional[int] = 51,
    pass_zero: Optional[Union[bool, str]] = True,
) -> np.ndarray:
    """Determines the filter coefficients for FIR filter.
    Mirrors `scipy.signal.firwin` and is cached so we don't have to design
    the filter each time it is applied.
    Args:
        cutoff_hz: Cut-off frequencies of filter [Hz]
        sampling_rate_hz: Sample Rate of data [Hz]
        num_taps: Filter order or number of filter coefficients
        pass_zero: If True, the gain at the frequency 0 (i.e., the “DC gain”) is 1.
                   If False, the DC gain is 0. Can also be a string argument for
                   the desired filter type: {'lowpass', 'highpass', 'bandpass', 'bandstop'}.
    Returns:
        Coefficients for the FIR filter (with length num_taps)
    """
    return scipy.signal.firwin(num_taps, cutoff_hz, pass_zero=pass_zero, fs=sampling_rate_hz).reshape(-1, 1)

class Filter:
    """Applies an FIR filter to each signal channel."""

    def __init__(
        self,
        num_taps: int,
        cutoff_hz: Union[float, List[float]],
        pass_zero: Union[bool, str],
    ) -> None:
        """Initializes general FIR filter class.
        Args:
            cutoff_hz: Cut-off frequencies of filter [Hz]
            sampling_rate_hz: Sample Rate of data [Hz]
            num_taps: Filter order or number of filter coefficients
            pass_zero: If True, the gain at the frequency 0 (i.e., the “DC gain”) is 1.
                       If False, the DC gain is 0. Can also be a string argument for
                       the desired filter type: {'lowpass', 'highpass', 'bandpass', 'bandstop'}.
        """
        super().__init__()
        self.num_taps = num_taps
        self.cutoff_hz = tuple(cutoff_hz) if isinstance(cutoff_hz, list) else cutoff_hz
        self.pass_zero = pass_zero

    def __call__(self, data):
        # Design filter
        filter = design_filter(self.cutoff_hz, 100, self.num_taps, self.pass_zero)

        # Apply filter to sensor stream
        data = scipy.signal.convolve(data, filter, mode="same")
        return torch.Tensor(data).double()

class Highpass(Filter):
    """Applies a high-pass FIR filter to each signal channel"""

    def __init__(
        self,
        num_taps: int,
        cutoff_hz: float,
    ) -> None:
        """Initializes high-pass filter class.
        Args:
            num_taps: Filter order or number of filter coefficients
            cutoff_hz: Cutoff frequency of the filter
            apply_to: List of modalities to apply transform to
        """
        super().__init__(num_taps, cutoff_hz, "highpass")

    def __repr__(self) -> str:
        return self.repr_helper(f"cutoff={self.cutoff_hz}-fs Hz, order={self.num_taps}")

class Lowpass(Filter):
    """Applies a low-pass FIR filter to each signal channel"""

    def __init__(
        self,
        num_taps: int,
        cutoff_hz: float,
    ) -> None:
        """Initializes low-pass filter class.
        Args:
            num_taps: Filter order or number of filter coefficients
            cutoff_hz: Cutoff frequency of the filter
            apply_to: List of modalities to apply transform to
        """
        super().__init__(num_taps, cutoff_hz, "lowpass")

    def __repr__(self) -> str:
        return self.repr_helper(f"0-{self.cutoff_hz} Hz, order={self.num_taps}")
    
