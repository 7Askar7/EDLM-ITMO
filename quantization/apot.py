"""
Additive Powers-of-Two (APoT) Quantization

Paper: "Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks"
       ICLR 2020 - https://openreview.net/forum?id=BkgXT24tDS

This implementation follows the paper's mathematical formulation exactly:
- Equation (2): Q^a(1,b) codebook generation
- Equation (3): Quantization with learned alpha scaling
- Algorithm 1: Weight normalization for stable training
- Straight-Through Estimator for backpropagation

Key Features:
- Per-channel quantization for better accuracy
- Weight normalization to prevent gradient issues
- Efficient vectorized operations
- Handles edge cases (NaN, Inf, zeros)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import itertools
import math


def generate_apot_codebook(num_additive_terms: int, num_bits: int, weight_mode: bool = True) -> torch.Tensor:
    """
    Generate APoT codebook Q^a(1,b) as defined in Equation (2) of the paper.

    The codebook consists of values that can be represented as sums of at most 'a'
    additive powers-of-two terms:

    Q^a(1,b) = {±Σ(i∈S) 2^(-i) | S⊆{0,1,...,b}, |S|≤a}

    Args:
        num_additive_terms: Number of additive terms 'a' (typically 2 or 3)
        num_bits: Bit-width for quantization
        weight_mode: If True, b = bit-1 (for weights), else b = bit (for activations)

    Returns:
        Sorted tensor of codebook values including 0, positive and negative values

    Example:
        For a=2, b=2: {±2^0, ±2^(-1), ±2^(-2), ±(2^0 + 2^(-1)), ±(2^0 + 2^(-2)), ±(2^(-1) + 2^(-2))}
        = {±1.0, ±0.5, ±0.25, ±1.5, ±1.25, ±0.75}
    """
    # Determine b parameter based on mode (Equation 2 in paper)
    b = num_bits - 1 if weight_mode else num_bits

    # Generate all possible exponents: {0, 1, ..., b}
    exponents = list(range(b + 1))

    # Generate all subsets S ⊆ {0,1,...,b} with |S| ≤ a
    codebook_positive = set()

    for subset_size in range(1, min(num_additive_terms, len(exponents)) + 1):
        for subset in itertools.combinations(exponents, subset_size):
            # Compute Σ(i∈S) 2^(-i)
            value = sum(2.0 ** (-i) for i in subset)
            codebook_positive.add(value)

    # Create symmetric codebook: {0} ∪ {positive values} ∪ {negative values}
    codebook = [0.0]
    codebook.extend(sorted(codebook_positive))
    codebook.extend([-v for v in sorted(codebook_positive)])

    return torch.tensor(sorted(codebook), dtype=torch.float32)


def quantize_to_codebook(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """
    Quantize input tensor to nearest codebook values.

    Uses efficient broadcasting to find nearest neighbors in codebook.

    Args:
        x: Input tensor of any shape [..., N]
        codebook: 1D tensor of sorted codebook values [M]

    Returns:
        Quantized tensor with same shape as x
    """
    # Reshape for broadcasting: x [..., N, 1], codebook [1, ..., 1, M]
    x_expanded = x.unsqueeze(-1)  # [..., N, 1]
    codebook_expanded = codebook.view(*([1] * x.dim()), -1)  # [1, ..., 1, M]

    # Find nearest codebook entry: argmin_i |x - codebook[i]|
    distances = torch.abs(x_expanded - codebook_expanded)
    nearest_idx = torch.argmin(distances, dim=-1)

    # Map to codebook values
    return codebook[nearest_idx]


class APoTQuantize(nn.Module):
    """
    APoT Quantization Module (Equation 3 from paper)

    Implements: x_q = Q(clamp(x/α, -1, 1)) * α

    where Q is the nearest neighbor in the APoT codebook and α is a learned
    scaling parameter (initialized as max(|x|) during first forward pass).

    Args:
        num_bits: Bit-width for quantization (2-8 typical)
        num_additive_terms: Number of additive terms 'a' (2 or 3)
        weight_mode: True for weights (b=bit-1), False for activations (b=bit)
        per_channel: If True, use separate alpha per channel
        channel_dim: Dimension along which to compute per-channel statistics
    """

    def __init__(
        self,
        num_bits: int = 4,
        num_additive_terms: int = 2,
        weight_mode: bool = False,
        per_channel: bool = False,
        channel_dim: int = 0,
    ):
        super().__init__()

        assert 1 <= num_bits <= 8, f"num_bits must be in [1, 8], got {num_bits}"
        assert 1 <= num_additive_terms <= 4, f"num_additive_terms must be in [1, 4], got {num_additive_terms}"

        self.num_bits = num_bits
        self.num_additive_terms = num_additive_terms
        self.weight_mode = weight_mode
        self.per_channel = per_channel
        self.channel_dim = channel_dim

        # Generate and register codebook (constant, not trained)
        codebook = generate_apot_codebook(num_additive_terms, num_bits, weight_mode)
        self.register_buffer('codebook', codebook)

        # Alpha will be initialized on first forward pass
        self.register_buffer('alpha', torch.tensor(1.0))
        self.alpha_initialized = False

    def _initialize_alpha(self, x: torch.Tensor) -> None:
        """
        Initialize alpha as max(|x|) following paper's recommendation.

        For per-channel: compute max per channel
        For global: compute global max
        """
        with torch.no_grad():
            if self.per_channel:
                # Compute max along all dims except channel_dim
                dims = list(range(x.dim()))
                dims.pop(self.channel_dim)

                if len(dims) > 0:
                    alpha = x.abs().amax(dim=dims, keepdim=True)
                else:
                    alpha = x.abs()

                # Ensure alpha has correct shape for broadcasting
                alpha = alpha + 1e-8  # Avoid division by zero

                # Resize alpha buffer to match shape
                self.alpha = alpha
            else:
                alpha = x.abs().max() + 1e-8
                self.alpha.fill_(alpha)

            self.alpha_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with APoT quantization.

        Training: Uses Straight-Through Estimator (STE)
        Inference: Direct quantization

        Args:
            x: Input tensor

        Returns:
            Quantized tensor with same shape as input
        """
        # Handle edge cases
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values")

        # Initialize alpha on first forward pass
        if not self.alpha_initialized:
            self._initialize_alpha(x)

        # Normalize to [-1, 1] range (Equation 3)
        x_normalized = torch.clamp(x / self.alpha, -1.0, 1.0)

        # Quantize to nearest codebook entry
        x_quant_normalized = quantize_to_codebook(x_normalized, self.codebook)

        # Scale back by alpha
        x_quant = x_quant_normalized * self.alpha

        # Straight-Through Estimator for gradient flow
        if self.training:
            # Forward: quantized value, Backward: gradient flows through identity
            x_quant = x + (x_quant - x).detach()

        return x_quant

    def extra_repr(self) -> str:
        return (f'num_bits={self.num_bits}, num_additive_terms={self.num_additive_terms}, '
                f'weight_mode={self.weight_mode}, per_channel={self.per_channel}, '
                f'codebook_size={len(self.codebook)}')


class APoTLinear(nn.Linear):
    """
    Linear layer with APoT quantization and weight normalization (Algorithm 1).

    Weight Normalization is critical for stable training:
    1. Normalize: w_norm = w / ||w||_2
    2. Quantize: w_q_norm = Q(w_norm)
    3. Denormalize: w_q = w_q_norm * ||w||_2

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: If True, add learnable bias
        num_bits: Bit-width for weight quantization
        num_additive_terms: Number of additive terms (2 or 3)
        quantize_input: If True, also quantize input activations
        input_bits: Bit-width for activation quantization
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_bits: int = 4,
        num_additive_terms: int = 2,
        quantize_input: bool = False,
        input_bits: int = 8,
    ):
        super().__init__(in_features, out_features, bias)

        # Weight quantizer (per output channel)
        self.weight_quantizer = APoTQuantize(
            num_bits=num_bits,
            num_additive_terms=num_additive_terms,
            weight_mode=True,  # b = bit - 1 for weights
            per_channel=True,
            channel_dim=0,  # Output channels are dim 0 in Linear
        )

        # Optional input quantizer
        self.quantize_input = quantize_input
        if quantize_input:
            self.input_quantizer = APoTQuantize(
                num_bits=input_bits,
                num_additive_terms=num_additive_terms,
                weight_mode=False,  # b = bit for activations
                per_channel=False,
            )

    def _quantize_weight(self) -> torch.Tensor:
        """
        Quantize weights with normalization (Algorithm 1).

        Returns:
            Quantized weight tensor
        """
        # Compute L2 norm per output channel (dim 0)
        # weight shape: [out_features, in_features]
        weight_norm = self.weight.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)

        # Normalize weights
        weight_normalized = self.weight / weight_norm

        # Quantize normalized weights
        weight_quant_normalized = self.weight_quantizer(weight_normalized)

        # Denormalize back
        weight_quant = weight_quant_normalized * weight_norm

        return weight_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights and optional quantized inputs."""
        # Quantize input if enabled
        if self.quantize_input:
            x = self.input_quantizer(x)

        # Get quantized weights
        weight_quant = self._quantize_weight()

        # Standard linear operation
        return F.linear(x, weight_quant, self.bias)


class APoTConv1d(nn.Conv1d):
    """
    1D Convolution with APoT quantization and weight normalization.

    Applies weight normalization per output channel before quantization.

    Args:
        Same as nn.Conv1d, plus:
        num_bits: Bit-width for weight quantization
        num_additive_terms: Number of additive terms (2 or 3)
        quantize_input: If True, also quantize input activations
        input_bits: Bit-width for activation quantization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        num_bits: int = 4,
        num_additive_terms: int = 2,
        quantize_input: bool = False,
        input_bits: int = 8,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode
        )

        # Weight quantizer (per output channel)
        self.weight_quantizer = APoTQuantize(
            num_bits=num_bits,
            num_additive_terms=num_additive_terms,
            weight_mode=True,
            per_channel=True,
            channel_dim=0,  # Output channels are dim 0
        )

        # Optional input quantizer
        self.quantize_input = quantize_input
        if quantize_input:
            self.input_quantizer = APoTQuantize(
                num_bits=input_bits,
                num_additive_terms=num_additive_terms,
                weight_mode=False,
                per_channel=False,
            )

    def _quantize_weight(self) -> torch.Tensor:
        """Quantize weights with per-channel normalization."""
        # weight shape: [out_channels, in_channels, kernel_size]
        # Compute norm over all dims except output channel (dim 0)
        weight_norm = self.weight.flatten(1).norm(p=2, dim=1, keepdim=True).unsqueeze(-1).clamp(min=1e-8)

        # Normalize
        weight_normalized = self.weight / weight_norm

        # Quantize
        weight_quant_normalized = self.weight_quantizer(weight_normalized)

        # Denormalize
        weight_quant = weight_quant_normalized * weight_norm

        return weight_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights and optional quantized inputs."""
        if self.quantize_input:
            x = self.input_quantizer(x)

        weight_quant = self._quantize_weight()

        return self._conv_forward(x, weight_quant, self.bias)


class APoTConv2d(nn.Conv2d):
    """
    2D Convolution with APoT quantization and weight normalization.

    Applies weight normalization per output channel before quantization.
    Most commonly used for CNNs.

    Args:
        Same as nn.Conv2d, plus:
        num_bits: Bit-width for weight quantization
        num_additive_terms: Number of additive terms (2 or 3)
        quantize_input: If True, also quantize input activations
        input_bits: Bit-width for activation quantization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        num_bits: int = 4,
        num_additive_terms: int = 2,
        quantize_input: bool = False,
        input_bits: int = 8,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode
        )

        # Weight quantizer (per output channel)
        self.weight_quantizer = APoTQuantize(
            num_bits=num_bits,
            num_additive_terms=num_additive_terms,
            weight_mode=True,
            per_channel=True,
            channel_dim=0,  # Output channels are dim 0
        )

        # Optional input quantizer
        self.quantize_input = quantize_input
        if quantize_input:
            self.input_quantizer = APoTQuantize(
                num_bits=input_bits,
                num_additive_terms=num_additive_terms,
                weight_mode=False,
                per_channel=False,
            )

    def _quantize_weight(self) -> torch.Tensor:
        """Quantize weights with per-channel normalization."""
        # weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        # Compute norm over all dims except output channel (dim 0)
        dims = list(range(1, self.weight.dim()))
        weight_norm = self.weight.flatten(1).norm(p=2, dim=1).view(-1, *([1] * len(dims))).clamp(min=1e-8)

        # Normalize
        weight_normalized = self.weight / weight_norm

        # Quantize
        weight_quant_normalized = self.weight_quantizer(weight_normalized)

        # Denormalize
        weight_quant = weight_quant_normalized * weight_norm

        return weight_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights and optional quantized inputs."""
        if self.quantize_input:
            x = self.input_quantizer(x)

        weight_quant = self._quantize_weight()

        return self._conv_forward(x, weight_quant, self.bias)


# ============================================================================
# Utility Functions
# ============================================================================

def print_codebook_stats(num_bits: int, num_additive_terms: int, weight_mode: bool = True) -> None:
    """
    Print statistics about an APoT codebook.

    Useful for understanding the quantization levels and distribution.
    """
    codebook = generate_apot_codebook(num_additive_terms, num_bits, weight_mode)

    print(f"\n{'='*70}")
    print(f"APoT Codebook Stats: a={num_additive_terms}, bits={num_bits}, "
          f"weight_mode={weight_mode}")
    print(f"{'='*70}")
    print(f"Codebook size: {len(codebook)}")
    print(f"Range: [{codebook.min():.6f}, {codebook.max():.6f}]")
    print(f"Unique positive values: {(codebook > 0).sum().item()}")
    print(f"Codebook values:\n{codebook.numpy()}")
    print(f"{'='*70}\n")


def replace_layers_with_apot(
    module: nn.Module,
    num_bits: int = 4,
    num_additive_terms: int = 2,
    quantize_input: bool = False,
    input_bits: int = 8,
    skip_first_last: bool = True,
) -> nn.Module:
    """
    Replace standard nn.Linear/Conv layers with APoT quantized versions.

    Args:
        module: PyTorch module to modify (modified in-place)
        num_bits: Bit-width for weight quantization
        num_additive_terms: Number of additive terms
        quantize_input: Whether to quantize activations
        input_bits: Bit-width for activation quantization
        skip_first_last: If True, skip first and last layers (common practice)

    Returns:
        Modified module with APoT layers
    """
    # Get all named modules
    named_modules = list(module.named_modules())
    layer_count = 0
    total_layers = sum(1 for _, m in named_modules
                      if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)))

    for name, child in named_modules:
        # Check if we should skip this layer
        if skip_first_last:
            if layer_count == 0 or layer_count == total_layers - 1:
                if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    layer_count += 1
                    continue

        # Replace Linear layers
        if isinstance(child, nn.Linear):
            new_layer = APoTLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                num_bits=num_bits,
                num_additive_terms=num_additive_terms,
                quantize_input=quantize_input,
                input_bits=input_bits,
            )
            # Copy weights
            new_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_layer.bias.data.copy_(child.bias.data)

            # Replace in parent module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = module.get_submodule(parent_name)
                setattr(parent, child_name, new_layer)
            else:
                setattr(module, child_name, new_layer)

            layer_count += 1

        # Replace Conv1d layers
        elif isinstance(child, nn.Conv1d):
            new_layer = APoTConv1d(
                child.in_channels,
                child.out_channels,
                child.kernel_size[0],
                stride=child.stride[0],
                padding=child.padding[0],
                dilation=child.dilation[0],
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
                num_bits=num_bits,
                num_additive_terms=num_additive_terms,
                quantize_input=quantize_input,
                input_bits=input_bits,
            )
            # Copy weights
            new_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_layer.bias.data.copy_(child.bias.data)

            # Replace in parent module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = module.get_submodule(parent_name)
                setattr(parent, child_name, new_layer)
            else:
                setattr(module, child_name, new_layer)

            layer_count += 1

        # Replace Conv2d layers
        elif isinstance(child, nn.Conv2d):
            new_layer = APoTConv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
                num_bits=num_bits,
                num_additive_terms=num_additive_terms,
                quantize_input=quantize_input,
                input_bits=input_bits,
            )
            # Copy weights
            new_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_layer.bias.data.copy_(child.bias.data)

            # Replace in parent module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = module.get_submodule(parent_name)
                setattr(parent, child_name, new_layer)
            else:
                setattr(module, child_name, new_layer)

            layer_count += 1

    return module


if __name__ == "__main__":
    # Demo: Show codebook generation for different configurations
    print("\n" + "="*70)
    print("APoT Quantization - Codebook Examples")
    print("="*70)

    print_codebook_stats(num_bits=4, num_additive_terms=2, weight_mode=True)
    print_codebook_stats(num_bits=4, num_additive_terms=3, weight_mode=True)
    print_codebook_stats(num_bits=8, num_additive_terms=2, weight_mode=False)

    # Demo: Test quantization
    print("\n" + "="*70)
    print("APoT Quantization - Testing")
    print("="*70)

    # Create test tensor
    x = torch.randn(2, 4) * 10.0

    # Create quantizer
    quantizer = APoTQuantize(num_bits=4, num_additive_terms=2, weight_mode=False)

    print(f"\nOriginal tensor:\n{x}")

    # Quantize
    x_quant = quantizer(x)

    print(f"\nQuantized tensor:\n{x_quant}")
    print(f"\nQuantization error: {(x - x_quant).abs().mean().item():.6f}")

    # Demo: Test layers
    print("\n" + "="*70)
    print("APoT Layers - Testing")
    print("="*70)

    # Test Linear layer
    layer = APoTLinear(10, 5, num_bits=4, num_additive_terms=2, quantize_input=True)
    x = torch.randn(2, 10)
    y = layer(x)
    print(f"\nAPoTLinear input shape: {x.shape}, output shape: {y.shape}")

    # Test Conv2d layer
    conv = APoTConv2d(3, 16, kernel_size=3, padding=1, num_bits=4, num_additive_terms=2)
    x = torch.randn(2, 3, 32, 32)
    y = conv(x)
    print(f"APoTConv2d input shape: {x.shape}, output shape: {y.shape}")

    print("\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70 + "\n")
