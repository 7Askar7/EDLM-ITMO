"""Fake quantization wrappers for neural network layers.

This module provides unified wrappers for applying different quantization
methods to neural network layers, supporting multiple quantization algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaround import AdaRoundQuantize
from .apot import APoTQuantize
from .dsq import DSQQuantize
from .lsq import LSQQuantizer
from .pact import PACTQuantizer


class FakeQuantize(nn.Module):
    """Wrapper for fake quantization of weights and activations.

    Provides a unified interface for different quantization methods,
    allowing easy switching between quantization algorithms.

    Args:
        quantizer_type: Type of quantizer to use. Options: 'lsq', 'pact',
            'adaround', 'apot', 'dsq' (default: 'lsq').
        bit_width: Number of bits for quantization (default: 8).
        symmetric: Use symmetric quantization (default: True).

    Attributes:
        weight_quantizer: Quantizer for weights.
        activation_quantizer: Quantizer for activations.

    Raises:
        ValueError: If quantizer_type is unknown.

    Example:
        >>> quantizer = FakeQuantize('lsq', bit_width=8)
        >>> quantized_weight = quantizer.quantize_weight(weight)
        >>> quantized_act = quantizer.quantize_activation(activation)
    """

    def __init__(self, quantizer_type='lsq', bit_width=8, symmetric=True):
        """Initialize fake quantization wrapper.

        Args:
            quantizer_type: Type of quantizer ('lsq', 'pact', 'adaround',
                'apot', or 'dsq').
            bit_width: Number of bits for quantization.
            symmetric: Use symmetric quantization.

        Raises:
            ValueError: If quantizer_type is not recognized.
        """
        super().__init__()

        self.quantizer_type = quantizer_type
        self.bit_width = bit_width
        self.symmetric = symmetric

        # Create weight and activation quantizers based on type
        if quantizer_type == 'lsq':
            self.weight_quantizer = LSQQuantizer(bit_width, symmetric)
            self.activation_quantizer = LSQQuantizer(
                bit_width, symmetric, all_positive=True
            )
        elif quantizer_type == 'pact':
            self.weight_quantizer = PACTQuantizer(bit_width, symmetric)
            # Use symmetric=True for activations (general case)
            self.activation_quantizer = PACTQuantizer(
                bit_width, symmetric=True
            )
        elif quantizer_type == 'adaround':
            self.weight_quantizer = AdaRoundQuantize(
                bit_width, signed=symmetric
            )
            self.activation_quantizer = AdaRoundQuantize(
                bit_width, signed=symmetric
            )
        elif quantizer_type == 'apot':
            self.weight_quantizer = APoTQuantize(
                num_bits=bit_width, weight_mode=True
            )
            self.activation_quantizer = APoTQuantize(
                num_bits=bit_width, weight_mode=False
            )
        elif quantizer_type == 'dsq':
            self.weight_quantizer = DSQQuantize(bit_width, signed=symmetric)
            self.activation_quantizer = DSQQuantize(
                bit_width, signed=symmetric
            )
        else:
            raise ValueError(f"Unknown quantizer type: {quantizer_type}")

    def quantize_weight(self, weight):
        """Quantize weight tensor.

        Args:
            weight: Weight tensor to quantize.

        Returns:
            Quantized weight tensor.
        """
        return self.weight_quantizer(weight)

    def quantize_activation(self, activation):
        """Quantize activation tensor.

        Args:
            activation: Activation tensor to quantize.

        Returns:
            Quantized activation tensor.
        """
        return self.activation_quantizer(activation)


class QuantizedLinear(nn.Linear):
    """
    Linear layer with quantization-aware training

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: If True, adds a learnable bias
        quantizer_type: Type of quantizer ('lsq' or 'pact')
        bit_width: Number of bits for quantization
        quantize_weight: Whether to quantize weights
        quantize_activation: Whether to quantize activations
    """

    def __init__(self, in_features, out_features, bias=True,
                 quantizer_type='lsq', bit_width=8,
                 quantize_weight=True, quantize_activation=True):
        super().__init__(in_features, out_features, bias)

        self.quantize_weight_flag = quantize_weight
        self.quantize_activation_flag = quantize_activation

        # Create quantizers
        if quantize_weight:
            if quantizer_type == 'lsq':
                self.weight_quantizer = LSQQuantizer(bit_width, symmetric=True)
            elif quantizer_type == 'pact':
                self.weight_quantizer = PACTQuantizer(bit_width, symmetric=True)
            elif quantizer_type == 'adaround':
                self.weight_quantizer = AdaRoundQuantize(bit_width, signed=True)
            elif quantizer_type == 'apot':
                self.weight_quantizer = APoTQuantize(num_bits=bit_width, weight_mode=True)
            elif quantizer_type == 'dsq':
                self.weight_quantizer = DSQQuantize(bit_width, signed=True)

        if quantize_activation:
            if quantizer_type == 'lsq':
                self.activation_quantizer = LSQQuantizer(bit_width, symmetric=True, all_positive=True)
            elif quantizer_type == 'pact':
                # Use symmetric=True for Linear layer outputs (can be negative before activation)
                self.activation_quantizer = PACTQuantizer(bit_width, symmetric=True)
            elif quantizer_type == 'adaround':
                self.activation_quantizer = AdaRoundQuantize(bit_width, signed=True)
            elif quantizer_type == 'apot':
                self.activation_quantizer = APoTQuantize(num_bits=bit_width, weight_mode=False)
            elif quantizer_type == 'dsq':
                self.activation_quantizer = DSQQuantize(bit_width, signed=True)

    def forward(self, x):
        # Quantize weight
        if self.quantize_weight_flag:
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight

        # Linear operation
        output = F.linear(x, weight, self.bias)

        # Quantize activation
        if self.quantize_activation_flag:
            output = self.activation_quantizer(output)

        return output


class QuantizedConv2d(nn.Conv2d):
    """
    Conv2d layer with quantization-aware training

    Args:
        quantizer_type: Type of quantizer ('lsq' or 'pact')
        bit_width: Number of bits for quantization
        quantize_weight: Whether to quantize weights
        quantize_activation: Whether to quantize activations
        activation_symmetric: Force symmetric activation quantization (useful for layers without ReLU)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 quantizer_type='lsq', bit_width=8,
                 quantize_weight=True, quantize_activation=True,
                 activation_symmetric=None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias)

        self.quantize_weight_flag = quantize_weight
        self.quantize_activation_flag = quantize_activation
        self.activation_symmetric = activation_symmetric

        # Create quantizers
        if quantize_weight:
            if quantizer_type == 'lsq':
                self.weight_quantizer = LSQQuantizer(bit_width, symmetric=True)
            elif quantizer_type == 'pact':
                self.weight_quantizer = PACTQuantizer(bit_width, symmetric=True)
            elif quantizer_type == 'adaround':
                self.weight_quantizer = AdaRoundQuantize(bit_width, signed=True)
            elif quantizer_type == 'apot':
                self.weight_quantizer = APoTQuantize(num_bits=bit_width, weight_mode=True)
            elif quantizer_type == 'dsq':
                self.weight_quantizer = DSQQuantize(bit_width, signed=True)

        if quantize_activation:
            if quantizer_type == 'lsq':
                self.activation_quantizer = LSQQuantizer(bit_width, symmetric=True, all_positive=True)
            elif quantizer_type == 'pact':
                # Default to symmetric=True for pre-ReLU activations (can be negative)
                # Only use asymmetric (False) when explicitly set for post-ReLU activations
                symmetric = True if self.activation_symmetric is None else self.activation_symmetric
                self.activation_quantizer = PACTQuantizer(bit_width, symmetric=symmetric)
            elif quantizer_type == 'adaround':
                self.activation_quantizer = AdaRoundQuantize(bit_width, signed=True)
            elif quantizer_type == 'apot':
                self.activation_quantizer = APoTQuantize(num_bits=bit_width, weight_mode=False)
            elif quantizer_type == 'dsq':
                self.activation_quantizer = DSQQuantize(bit_width, signed=True)

    def forward(self, x):
        # Quantize weight
        if self.quantize_weight_flag:
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight

        # Conv operation
        output = F.conv2d(x, weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)

        # Quantize activation
        if self.quantize_activation_flag:
            output = self.activation_quantizer(output)

        return output


class QuantizedLSTMCell(nn.Module):
    """
    LSTM cell with quantization-aware training

    Args:
        input_size: Size of input features
        hidden_size: Size of hidden state
        quantizer_type: Type of quantizer ('lsq' or 'pact')
        bit_width: Number of bits for quantization
    """

    def __init__(self, input_size, hidden_size,
                 quantizer_type='lsq', bit_width=8):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM weights
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

        # Initialize weights
        self.reset_parameters()

        # Create quantizers
        if quantizer_type == 'lsq':
            self.weight_quantizer_ih = LSQQuantizer(bit_width, symmetric=True)
            self.weight_quantizer_hh = LSQQuantizer(bit_width, symmetric=True)
            self.activation_quantizer = LSQQuantizer(bit_width, symmetric=True)
        elif quantizer_type == 'pact':
            self.weight_quantizer_ih = PACTQuantizer(bit_width, symmetric=True)
            self.weight_quantizer_hh = PACTQuantizer(bit_width, symmetric=True)
            # Use symmetric=True for LSTM activations (tanh outputs can be negative)
            self.activation_quantizer = PACTQuantizer(bit_width, symmetric=True)
        elif quantizer_type == 'adaround':
            self.weight_quantizer_ih = AdaRoundQuantize(bit_width, signed=True)
            self.weight_quantizer_hh = AdaRoundQuantize(bit_width, signed=True)
            self.activation_quantizer = AdaRoundQuantize(bit_width, signed=True)
        elif quantizer_type == 'apot':
            self.weight_quantizer_ih = APoTQuantize(bit_width, signed=True)
            self.weight_quantizer_hh = APoTQuantize(bit_width, signed=True)
            self.activation_quantizer = APoTQuantize(num_bits=bit_width, weight_mode=False)
        elif quantizer_type == 'dsq':
            self.weight_quantizer_ih = DSQQuantize(bit_width, signed=True)
            self.weight_quantizer_hh = DSQQuantize(bit_width, signed=True)
            self.activation_quantizer = DSQQuantize(bit_width, signed=True)

    def reset_parameters(self):
        """Initialize parameters"""
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden=None):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, input_size)
            hidden: Tuple of (h, c) with shape (batch, hidden_size)

        Returns:
            h_next: Next hidden state
            c_next: Next cell state
        """
        batch_size = x.size(0)

        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c = hidden

        # Quantize weights
        weight_ih = self.weight_quantizer_ih(self.weight_ih)
        weight_hh = self.weight_quantizer_hh(self.weight_hh)

        # LSTM computation
        gates = (torch.mm(x, weight_ih.t()) + self.bias_ih +
                 torch.mm(h, weight_hh.t()) + self.bias_hh)

        # Quantize gates
        gates = self.activation_quantizer(gates)

        # Split gates
        i, f, g, o = gates.chunk(4, 1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class QuantizedConv1d(nn.Module):
    """
    Conv1d layer with fake quantization
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 quantizer_type='lsq', bit_width=8):
        super(QuantizedConv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias)

        # Create weight and activation quantizers
        if quantizer_type == 'lsq':
            self.weight_quantizer = LSQQuantizer(bit_width, symmetric=True)
            self.activation_quantizer = LSQQuantizer(bit_width, symmetric=False, all_positive=True)
        elif quantizer_type == 'pact':
            self.weight_quantizer = PACTQuantizer(bit_width, symmetric=True)
            # Use symmetric=True for pre-ReLU activations (can be negative)
            self.activation_quantizer = PACTQuantizer(bit_width, symmetric=True)
        elif quantizer_type == 'adaround':
            self.weight_quantizer = AdaRoundQuantize(bit_width, signed=True)
            self.activation_quantizer = AdaRoundQuantize(bit_width, signed=True)
        elif quantizer_type == 'apot':
            self.weight_quantizer = APoTQuantize(num_bits=bit_width, weight_mode=True)
            self.activation_quantizer = APoTQuantize(num_bits=bit_width, weight_mode=False)
        elif quantizer_type == 'dsq':
            self.weight_quantizer = DSQQuantize(bit_width, signed=True)
            self.activation_quantizer = DSQQuantize(bit_width, signed=True)
        else:
            raise ValueError(f"Unknown quantizer type: {quantizer_type}")

    def forward(self, x):
        # Quantize input activations
        x_q = self.activation_quantizer(x)

        # Quantize weights
        weight_q = self.weight_quantizer(self.conv.weight)

        # Perform convolution with quantized values
        output = F.conv1d(x_q, weight_q, self.conv.bias,
                         self.conv.stride, self.conv.padding,
                         self.conv.dilation, self.conv.groups)

        return output


def replace_layers_with_quantized(model, quantizer_type='lsq', bit_width=8):
    """Replace standard layers with quantized versions.

    Recursively replaces nn.Linear and nn.Conv2d layers in a model with
    their quantized equivalents. Weights and biases are copied from the
    original layers.

    Args:
        model: PyTorch model to modify (modified in-place).
        quantizer_type: Type of quantizer to use ('lsq', 'pact', 'adaround',
            'apot', or 'dsq') (default: 'lsq').
        bit_width: Number of bits for quantization (default: 8).

    Returns:
        Modified model with quantized layers.

    Example:
        >>> model = torchvision.models.resnet18()
        >>> quantized_model = replace_layers_with_quantized(
        ...     model, quantizer_type='lsq', bit_width=8
        ... )

    Note:
        This function modifies the model in-place and also returns it.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace Linear with QuantizedLinear
            new_module = QuantizedLinear(
                module.in_features,
                module.out_features,
                module.bias is not None,
                quantizer_type=quantizer_type,
                bit_width=bit_width
            )
            new_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.clone()
            setattr(model, name, new_module)

        elif isinstance(module, nn.Conv2d):
            # Replace Conv2d with QuantizedConv2d
            new_module = QuantizedConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
                quantizer_type=quantizer_type,
                bit_width=bit_width
            )
            new_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.clone()
            setattr(model, name, new_module)

        else:
            # Recursively replace in child modules
            replace_layers_with_quantized(module, quantizer_type, bit_width)

    return model
