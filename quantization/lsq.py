"""LSQ (Learned Step Size Quantization) implementation.

Reference: "Learned Step Size Quantization" (ICLR 2020)
Paper: https://arxiv.org/abs/1902.08153

This module implements LSQ and LSQ+ quantization methods, which learn
the quantization step size during training for improved accuracy.
"""

import numpy as np
import torch
import torch.nn as nn

from .base import BaseQuantizer, round_ste


class LSQQuantizer(BaseQuantizer):
    """Learned Step Size Quantization (LSQ).

    LSQ learns the quantization step size during training, which allows
    for better accuracy compared to fixed quantization schemes. The step
    size is a learnable parameter that is optimized along with the network
    weights.

    Args:
        bit_width: Number of bits for quantization (default: 8).
        symmetric: Use symmetric quantization (default: True).
        per_channel: Use per-channel quantization (default: False).
        all_positive: Assume all values are positive, e.g., after ReLU
            (default: False).

    Attributes:
        step_size: Learnable parameter for quantization step size.
        initialized: Buffer tracking whether step size has been initialized.
    """

    def __init__(
        self,
        bit_width=8,
        symmetric=True,
        per_channel=False,
        all_positive=False
    ):
        """Initialize LSQ quantizer.

        Args:
            bit_width: Number of bits for quantization.
            symmetric: Use symmetric quantization.
            per_channel: Use per-channel quantization.
            all_positive: Assume all values are positive.
        """
        super().__init__(bit_width, symmetric)
        self.per_channel = per_channel
        self.all_positive = all_positive

        # Initialize step size as learnable parameter
        self.register_buffer('initialized', torch.tensor(0))
        self.step_size = nn.Parameter(torch.ones(1))

    def initialize_step_size(self, x):
        """Initialize step size based on input statistics.

        The initialization strategy differs based on whether inputs are
        all positive (e.g., activations after ReLU) or can be negative
        (e.g., weights).

        Args:
            x: Input tensor used for initialization.
        """
        if self.initialized == 0:
            if self.all_positive:
                # For activations (ReLU), use mean-based initialization
                init_val = (
                    x.detach().abs().mean() * 2 / (self.qmax ** 0.5)
                )
            else:
                # For weights, use standard deviation-based initialization
                init_val = (
                    2 * x.detach().abs().mean() / (self.qmax ** 0.5)
                )

            self.step_size.data.fill_(init_val)
            self.initialized.fill_(1)

    def grad_scale(self, x, scale):
        """Compute gradient scaling factor for LSQ.

        The gradient scaling factor helps stabilize training by normalizing
        gradients based on tensor size and quantization range.

        Args:
            x: Input tensor.
            scale: Current scale value.

        Returns:
            Gradient scaling factor.
        """
        n = x.numel()
        # Gradient scaling factor from LSQ paper
        grad_scale_factor = 1.0 / np.sqrt(n * self.qmax)
        return grad_scale_factor

    def quantize(self, x):
        """Quantize input tensor.

        Applies quantization by normalizing, clamping, and rounding to
        discrete levels.

        Args:
            x: Input tensor to quantize.

        Returns:
            Quantized tensor in discrete levels.
        """
        # Normalize by step size
        x_scaled = x / self.step_size

        # Clamp to quantization range
        x_clamped = torch.clamp(x_scaled, self.qmin, self.qmax)

        # Round with straight-through estimator
        x_quantized = round_ste(x_clamped)

        return x_quantized

    def dequantize(self, x_q, scale=None):
        """Dequantize tensor back to floating point.

        Args:
            x_q: Quantized tensor in discrete levels.
            scale: Scaling factor (uses self.step_size if None).

        Returns:
            Dequantized floating point tensor.
        """
        if scale is None:
            scale = self.step_size
        return x_q * scale

    def forward(self, x):
        """Forward pass with LSQ quantization.

        Performs quantization and dequantization with learned step size.
        During training, applies gradient scaling for stable optimization.

        Args:
            x: Input tensor.

        Returns:
            Quantized and dequantized tensor.
        """
        if self.training:
            self.initialize_step_size(x)

        # Apply gradient scaling to step size for stable training
        grad_scale_factor = self.grad_scale(x, self.step_size)
        step_size_scaled = (
            self.step_size * grad_scale_factor
            + self.step_size * (1 - grad_scale_factor)
        )

        # Quantize
        x_q = self.quantize(x)

        # Dequantize
        x_dq = self.dequantize(x_q, step_size_scaled)

        return x_dq


class LSQPlusQuantizer(LSQQuantizer):
    """LSQ+ variant with separate positive and negative step sizes.

    LSQ+ extends LSQ by learning separate step sizes for positive and
    negative values, which can improve quantization accuracy for data
    with asymmetric distributions.

    Args:
        bit_width: Number of bits for quantization (default: 8).
        symmetric: Use symmetric quantization (default: True).
        per_channel: Use per-channel quantization (default: False).

    Attributes:
        step_size_pos: Learnable step size for positive values.
        step_size_neg: Learnable step size for negative values.
    """

    def __init__(self, bit_width=8, symmetric=True, per_channel=False):
        """Initialize LSQ+ quantizer.

        Args:
            bit_width: Number of bits for quantization.
            symmetric: Use symmetric quantization.
            per_channel: Use per-channel quantization.
        """
        super().__init__(bit_width, symmetric, per_channel, all_positive=False)

        # Separate step sizes for positive and negative values
        self.step_size_pos = nn.Parameter(torch.ones(1))
        self.step_size_neg = nn.Parameter(torch.ones(1))

    def initialize_step_size(self, x):
        """Initialize separate step sizes for positive and negative values.

        Computes statistics separately for positive and negative values
        to initialize their respective step sizes.

        Args:
            x: Input tensor used for initialization.
        """
        if self.initialized == 0:
            x_pos = x.detach()[x > 0]
            x_neg = x.detach()[x < 0]

            if len(x_pos) > 0:
                init_val_pos = (
                    x_pos.abs().mean() * 2 / (self.qmax ** 0.5)
                )
                self.step_size_pos.data.fill_(init_val_pos)

            if len(x_neg) > 0:
                init_val_neg = (
                    x_neg.abs().mean() * 2 / (abs(self.qmin) ** 0.5)
                )
                self.step_size_neg.data.fill_(init_val_neg)

            self.initialized.fill_(1)

    def forward(self, x):
        """Forward pass with LSQ+ quantization.

        Applies separate quantization for positive and negative values
        using their respective learned step sizes.

        Args:
            x: Input tensor.

        Returns:
            Quantized and dequantized tensor.
        """
        if self.training:
            self.initialize_step_size(x)

        # Split into positive and negative parts
        mask_pos = x >= 0
        mask_neg = x < 0

        x_out = torch.zeros_like(x)

        # Quantize positive values
        if mask_pos.any():
            x_pos = x[mask_pos]
            x_pos_scaled = x_pos / self.step_size_pos
            x_pos_clamped = torch.clamp(x_pos_scaled, 0, self.qmax)
            x_pos_q = round_ste(x_pos_clamped)
            x_pos_dq = x_pos_q * self.step_size_pos
            x_out[mask_pos] = x_pos_dq

        # Quantize negative values
        if mask_neg.any():
            x_neg = x[mask_neg]
            x_neg_scaled = x_neg / self.step_size_neg
            x_neg_clamped = torch.clamp(x_neg_scaled, self.qmin, 0)
            x_neg_q = round_ste(x_neg_clamped)
            x_neg_dq = x_neg_q * self.step_size_neg
            x_out[mask_neg] = x_neg_dq

        return x_out
