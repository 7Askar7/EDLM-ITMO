"""Base class for quantization methods.

This module provides abstract base classes and utilities for implementing
various quantization methods.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseQuantizer(ABC, nn.Module):
    """Abstract base class for quantization methods.

    This class provides common functionality for quantization methods,
    including quantization range calculation and abstract methods that
    subclasses must implement.

    Args:
        bit_width: Number of bits for quantization (default: 8).
        symmetric: Whether to use symmetric quantization range (default: True).
            If True, range is [-2^(n-1), 2^(n-1)-1].
            If False, range is [0, 2^n-1].
    """

    def __init__(self, bit_width=8, symmetric=True):
        """Initialize the base quantizer.

        Args:
            bit_width: Number of bits for quantization.
            symmetric: Whether to use symmetric quantization.
        """
        super().__init__()
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.n_levels = 2 ** bit_width

        # Calculate quantization range based on symmetric flag
        if symmetric:
            self.qmin = -(2 ** (bit_width - 1))
            self.qmax = 2 ** (bit_width - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bit_width - 1

    @abstractmethod
    def forward(self, x):
        """Forward pass with quantization.

        Args:
            x: Input tensor to quantize.

        Returns:
            Quantized tensor.
        """
        pass

    @abstractmethod
    def quantize(self, x):
        """Quantize tensor to discrete levels.

        Args:
            x: Input tensor to quantize.

        Returns:
            Quantized tensor in integer representation.
        """
        pass

    @abstractmethod
    def dequantize(self, x_q, scale):
        """Dequantize tensor back to floating point.

        Args:
            x_q: Quantized tensor in integer representation.
            scale: Scaling factor for dequantization.

        Returns:
            Dequantized tensor in floating point.
        """
        pass


class GradientPassThrough(torch.autograd.Function):
    """Straight-Through Estimator (STE) for gradient pass-through.

    This function allows gradients to pass through the rounding operation
    unchanged during backpropagation, which is essential for training
    quantized neural networks.

    The forward pass applies rounding, while the backward pass acts as
    an identity function for gradients.
    """

    @staticmethod
    def forward(ctx, x):
        """Forward pass: round input values.

        Args:
            ctx: Context object for saving information for backward pass.
            x: Input tensor.

        Returns:
            Rounded tensor.
        """
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: pass gradients through unchanged.

        Args:
            ctx: Context object with saved information from forward pass.
            grad_output: Gradient with respect to output.

        Returns:
            Gradient with respect to input (unchanged).
        """
        return grad_output


def round_ste(x):
    """Round with straight-through estimator.

    Convenience function that applies the GradientPassThrough function
    to round input values while allowing gradients to pass through.

    Args:
        x: Input tensor to round.

    Returns:
        Rounded tensor with gradient pass-through.
    """
    return GradientPassThrough.apply(x)
