"""PACT (Parameterized Clipping Activation) implementation.

Reference: "PACT: Parameterized Clipping Activation for Quantized Neural
Networks" (2018)
Paper: https://arxiv.org/abs/1805.06085

This module implements PACT quantization, which learns a clipping threshold
for activations during training to maintain accuracy in quantized networks.
"""

import torch
import torch.nn as nn

from .base import BaseQuantizer, round_ste


class PACTQuantizer(BaseQuantizer):
    """Parameterized Clipping Activation for Quantization (PACT).

    PACT learns a clipping threshold (alpha) for activations during training,
    which helps maintain accuracy in quantized networks. The learned threshold
    adapts to the data distribution.

    Args:
        bit_width: Number of bits for quantization (default: 8).
        symmetric: Use symmetric quantization (default: True).
        alpha_init: Initial value for clipping threshold (default: 10.0).
        learn_alpha: Whether to learn alpha parameter (default: True).

    Attributes:
        alpha: Learnable or fixed clipping threshold parameter.
        learn_alpha: Flag indicating if alpha is learnable.
        initialized: Buffer tracking alpha initialization status.
    """

    def __init__(
        self,
        bit_width=8,
        symmetric=True,
        alpha_init=10.0,
        learn_alpha=True
    ):
        """Initialize PACT quantizer.

        Args:
            bit_width: Number of bits for quantization.
            symmetric: Use symmetric quantization.
            alpha_init: Initial value for clipping threshold.
            learn_alpha: Whether to learn alpha parameter.
        """
        super().__init__(bit_width, symmetric)

        # Learnable clipping threshold
        if learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.register_buffer('alpha', torch.tensor(alpha_init))

        self.learn_alpha = learn_alpha

        # Register buffer for tracking initialization
        self.register_buffer('initialized', torch.tensor(0))

    def initialize_alpha(self, x):
        """Initialize alpha based on input statistics.

        Uses the maximum absolute value of input to initialize alpha,
        ensuring the clipping range covers the data distribution.

        Args:
            x: Input tensor used for initialization.
        """
        if self.initialized.item() == 0 and self.learn_alpha:
            # Initialize alpha to contain most of the data
            init_val = x.detach().abs().max()
            self.alpha.data.fill_(init_val)
            self.initialized.fill_(1)

    def clip_function(self, x):
        """Clipping function with gradient pass-through.

        Clips input values to [-alpha, alpha] for symmetric quantization
        or [0, alpha] for asymmetric quantization.

        Args:
            x: Input tensor to clip.

        Returns:
            Clipped tensor.
        """
        alpha_abs = self.alpha.abs()
        if self.symmetric:
            # Symmetric clipping: [-alpha, alpha]
            return torch.maximum(
                torch.minimum(x, alpha_abs), -alpha_abs
            )
        else:
            # Asymmetric clipping: [0, alpha]
            return torch.maximum(
                torch.minimum(x, alpha_abs),
                torch.tensor(0.0, device=x.device, dtype=x.dtype)
            )

    def get_scale(self):
        """Calculate quantization scale based on alpha.

        Returns:
            Quantization scale factor.
        """
        if self.symmetric:
            scale = 2 * self.alpha / (self.qmax - self.qmin)
        else:
            scale = self.alpha / (self.qmax - self.qmin)
        return scale

    def quantize(self, x):
        """Quantize input tensor with PACT.

        Applies clipping, normalization, and rounding to quantize input.

        Args:
            x: Input tensor to quantize.

        Returns:
            Tuple of (quantized tensor, scale factor).
        """
        # Apply clipping
        x_clipped = self.clip_function(x)

        # Calculate scale
        scale = self.get_scale()

        # Normalize
        x_scaled = x_clipped / scale

        # Shift if asymmetric
        if not self.symmetric:
            x_scaled = x_scaled - self.qmin

        # Round with straight-through estimator
        x_quantized = round_ste(x_scaled)

        # Clamp to quantization range
        x_quantized = torch.clamp(x_quantized, self.qmin, self.qmax)

        return x_quantized, scale

    def dequantize(self, x_q, scale):
        """Dequantize tensor back to floating point.

        Args:
            x_q: Quantized tensor.
            scale: Scaling factor for dequantization.

        Returns:
            Dequantized tensor.
        """
        # Shift back if asymmetric
        if not self.symmetric:
            x_q = x_q + self.qmin

        # Scale back
        x_dq = x_q * scale

        return x_dq

    def forward(self, x):
        """Forward pass with PACT quantization.

        Performs quantization and dequantization with learned alpha.
        During training, initializes alpha on first call.

        Args:
            x: Input tensor.

        Returns:
            Quantized and dequantized tensor.
        """
        if self.training:
            self.initialize_alpha(x)

        # Quantize
        x_q, scale = self.quantize(x)

        # Dequantize
        x_dq = self.dequantize(x_q, scale)

        return x_dq


class PACTActivation(nn.Module):
    """PACT activation function (learnable ReLU with upper bound).

    This is commonly used as a drop-in replacement for ReLU in quantized
    networks. It clips activations to [0, alpha] where alpha is learned.

    Args:
        alpha_init: Initial value for clipping threshold (default: 10.0).

    Attributes:
        alpha: Learnable upper bound parameter.
    """

    def __init__(self, alpha_init=10.0):
        """Initialize PACT activation.

        Args:
            alpha_init: Initial value for clipping threshold.
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x):
        """Apply PACT activation: clip(x, 0, alpha).

        Args:
            x: Input tensor.

        Returns:
            Clipped tensor in range [0, alpha].
        """
        alpha_abs = self.alpha.abs()
        return torch.maximum(
            torch.minimum(x, alpha_abs),
            torch.tensor(0.0, device=x.device, dtype=x.dtype)
        )


class PACTQuantizerV2(BaseQuantizer):
    """Improved PACT quantizer with better gradient handling.

    This version includes:
    - Better initialization strategy using percentiles
    - Gradient clipping for alpha to prevent instability
    - Per-channel quantization support

    Args:
        bit_width: Number of bits for quantization (default: 8).
        symmetric: Use symmetric quantization (default: True).
        alpha_init: Initial value for clipping threshold (default: 10.0).
        per_channel: Use per-channel quantization (default: False).
        num_channels: Number of channels for per-channel quantization
            (default: 1).

    Attributes:
        alpha: Learnable clipping threshold(s).
        per_channel: Flag for per-channel quantization.
        num_channels: Number of channels.
        initialized: Buffer tracking initialization status.
    """

    def __init__(
        self,
        bit_width=8,
        symmetric=True,
        alpha_init=10.0,
        per_channel=False,
        num_channels=1
    ):
        """Initialize improved PACT quantizer.

        Args:
            bit_width: Number of bits for quantization.
            symmetric: Use symmetric quantization.
            alpha_init: Initial value for clipping threshold.
            per_channel: Use per-channel quantization.
            num_channels: Number of channels.
        """
        super().__init__(bit_width, symmetric)

        self.per_channel = per_channel
        self.num_channels = num_channels

        # Learnable clipping threshold
        if per_channel:
            self.alpha = nn.Parameter(
                torch.ones(num_channels) * alpha_init
            )
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Register buffer for tracking initialization
        self.register_buffer('initialized', torch.tensor(0))

    def initialize_alpha(self, x):
        """Initialize alpha based on input statistics with better strategy.

        Uses 99th percentile instead of max for more robust initialization.

        Args:
            x: Input tensor for initialization.
        """
        if self.initialized.item() == 0:
            if self.per_channel:
                # Per-channel initialization
                # Assume x is [N, C, ...] format
                alpha_init = []
                for c in range(self.num_channels):
                    channel_data = x[:, c, ...].detach()
                    # Use 99th percentile for robustness
                    # Convert to float32 for quantile compatibility
                    percentile_val = torch.quantile(
                        channel_data.abs().float(), 0.99
                    )
                    alpha_init.append(percentile_val)
                self.alpha.data = torch.tensor(alpha_init, device=x.device)
            else:
                # Global initialization
                # Convert to float32 for quantile compatibility
                percentile_val = torch.quantile(
                    x.detach().abs().float(), 0.99
                )
                self.alpha.data.fill_(percentile_val)

            self.initialized.fill_(1)

    def _reshape_alpha(self):
        """Reshape alpha for broadcasting.

        Returns:
            Reshaped alpha tensor for per-channel operations.
        """
        if self.per_channel:
            return self.alpha.view(1, -1, 1, 1)
        return self.alpha

    def quantize(self, x):
        """Quantize input tensor with improved PACT.

        Args:
            x: Input tensor to quantize.

        Returns:
            Tuple of (quantized tensor, scale factor).
        """
        alpha = self._reshape_alpha()
        alpha_abs = alpha.abs()

        # Apply clipping based on symmetric flag
        if self.symmetric:
            x_clipped = torch.clamp(x, -alpha_abs, alpha_abs)
            scale = 2 * alpha_abs / (self.qmax - self.qmin)
        else:
            x_clipped = torch.clamp(x, torch.zeros_like(x), alpha_abs)
            scale = alpha_abs / (self.qmax - self.qmin)

        # Ensure scale is not too small
        scale = torch.clamp(scale, min=1e-8)

        # Normalize
        x_scaled = x_clipped / scale
        if not self.symmetric:
            x_scaled = x_scaled - self.qmin

        # Round and clamp
        x_q = round_ste(x_scaled)
        x_q = torch.clamp(x_q, self.qmin, self.qmax)

        return x_q, scale

    def dequantize(self, x_q, scale):
        """Dequantize tensor.

        Args:
            x_q: Quantized tensor.
            scale: Scaling factor.

        Returns:
            Dequantized tensor.
        """
        if not self.symmetric:
            x_q = x_q + self.qmin
        return x_q * scale

    def forward(self, x):
        """Forward pass with improved PACT quantization.

        Includes gradient clipping for alpha to prevent training instability.

        Args:
            x: Input tensor.

        Returns:
            Quantized and dequantized tensor.
        """
        if self.training:
            self.initialize_alpha(x)

            # Gradient clipping for alpha to prevent instability
            if self.alpha.grad is not None:
                self.alpha.grad = torch.clamp(
                    self.alpha.grad, -0.1, 0.1
                )

        x_q, scale = self.quantize(x)
        x_dq = self.dequantize(x_q, scale)

        return x_dq
