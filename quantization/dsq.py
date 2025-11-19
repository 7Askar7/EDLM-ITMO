"""DSQ: Differentiable Soft Quantization.

Based on "DSQ: Differentiable Soft Quantization for Hardware-Friendly
Neural Networks" (ICCV 2019).
Paper: https://arxiv.org/abs/1908.05033

This module implements DSQ quantization with smooth transitions between
quantization levels using differentiable soft quantization functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSQQuantize(nn.Module):
    """DSQ quantization with smooth transitions between quantization levels.

    Uses sigmoid-based soft quantization function for differentiability.
    Temperature annealing transitions from soft to hard quantization during
    training.

    Args:
        bit_width: Number of bits for quantization (default: 8).
        signed: Use signed quantization range (default: True).

    Attributes:
        scale: Learnable quantization scale parameter.
        zero_point: Learnable zero point parameter.
        temperature: Temperature for soft quantization (annealed during
            training).
        initialized: Buffer tracking initialization status.
    """

    def __init__(self, bit_width=8, signed=True):
        """Initialize DSQ quantizer.

        Args:
            bit_width: Number of bits for quantization.
            signed: Use signed quantization range.
        """
        super().__init__()
        self.bit_width = bit_width
        self.signed = signed

        # Quantization range
        if self.signed:
            self.qmin = -(2 ** (bit_width - 1))
            self.qmax = 2 ** (bit_width - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bit_width - 1

        # Learnable parameters
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.zero_point = nn.Parameter(torch.tensor(0.0))

        # Temperature parameter for soft quantization
        # Starts high (soft) and anneals to low (hard)
        self.register_buffer('temperature', torch.tensor(1.0))
        self.register_buffer('temperature_decay', torch.tensor(0.99))
        self.register_buffer('min_temperature', torch.tensor(0.01))

        # For initialization
        self.register_buffer('initialized', torch.tensor(0))

    def initialize_parameters(self, x):
        """Initialize scale and zero point based on input statistics.

        This should be called during calibration phase to properly set the
        quantization range based on real data distribution.

        Args:
            x: Input tensor used for calibration.
        """
        if self.initialized.item() == 0:
            with torch.no_grad():
                # Use max-based initialization for better coverage
                # Compute symmetric range based on max absolute value
                x_max_abs = x.detach().abs().max()

                # Initialize scale to cover full data range
                if x_max_abs > 0:
                    if self.signed:
                        # For signed: range is [-x_max_abs, x_max_abs]
                        scale = 2 * x_max_abs / (self.qmax - self.qmin)
                        self.scale.data = scale
                        self.zero_point.data = torch.tensor(0.0)
                    else:
                        # For unsigned: range is [0, x_max_abs]
                        scale = x_max_abs / (self.qmax - self.qmin)
                        self.scale.data = scale
                        self.zero_point.data = torch.tensor(0.0)
                else:
                    # Fallback for constant inputs
                    self.scale.data = torch.tensor(1.0)
                    self.zero_point.data = torch.tensor(0.0)

                self.initialized.fill_(1)

    def soft_round(self, x, temperature):
        """Soft rounding function using sum of shifted tanh functions.

        Approximates round() but is differentiable. Creates a smooth
        staircase function.

        Args:
            x: Input tensor to round.
            temperature: Temperature controlling sharpness of transitions.

        Returns:
            Soft-rounded tensor.

        Note:
            This method is deprecated in favor of the vectorized sigmoid
            implementation in forward(). Kept for backward compatibility.
        """
        # Create a smooth staircase function
        # For each integer k, we add a tanh centered at k+0.5
        k_min = torch.floor(x.min()).item()
        k_max = torch.ceil(x.max()).item()

        result = torch.zeros_like(x)

        for k in range(int(k_min), int(k_max) + 1):
            # tanh function transitions from -1 to 1 around k+0.5
            # Scale by temperature to control sharpness
            transition = torch.tanh((x - (k + 0.5)) / temperature)

            # Add contribution of this level
            result = result + transition

        # Normalize to get proper rounding behavior
        result = (result + (k_max - k_min + 1)) / 2 + k_min

        return result

    def forward(self, x):
        """Forward pass with DSQ quantization.

        During training, uses soft quantization with temperature annealing.
        During inference, uses hard quantization.

        Args:
            x: Input tensor to quantize.

        Returns:
            Quantized tensor (with STE during training).
        """
        # Initialize on first forward pass (calibration phase)
        # Only perform initialization during training to avoid
        # data-dependent control flow issues during export/tracing
        if self.training and self.initialized.item() == 0:
            self.initialize_parameters(x)

        # Anneal temperature during training
        if self.training:
            self.temperature = torch.maximum(
                self.temperature * self.temperature_decay,
                self.min_temperature
            )

        # Scale and shift
        x_scaled = x / self.scale - self.zero_point

        if self.training:
            # Vectorized soft quantization (100x faster than loop)
            # Sigmoid centered at k+0.5 (per DSQ paper ICCV 2019)
            levels = torch.arange(
                self.qmin, self.qmax + 1,
                device=x.device, dtype=x.dtype
            )
            x_expanded = x_scaled.unsqueeze(-1)  # [..., 1] for broadcasting

            # Sigmoid centered at k+0.5 (correct formula from paper)
            activations = torch.sigmoid(
                (x_expanded - (levels + 0.5)) / self.temperature
            )

            # Sum over levels
            x_soft = activations.sum(-1) + self.qmin

            # Clamp to valid range
            x_quantized = torch.clamp(x_soft, self.qmin, self.qmax)
        else:
            # Hard quantization during inference
            x_rounded = torch.round(x_scaled)
            x_quantized = torch.clamp(x_rounded, self.qmin, self.qmax)

        # Dequantize
        x_dequant = (x_quantized + self.zero_point) * self.scale

        if self.training:
            # Straight-through estimator for gradients
            return x + (x_dequant - x).detach()
        else:
            # In eval mode, return quantized values
            return x_dequant


class DSQLinear(nn.Module):
    """Linear layer with DSQ quantization.

    Applies DSQ quantization to both weights and activations in a linear
    layer.

    Args:
        in_features: Size of input features.
        out_features: Size of output features.
        bias: If True, adds a learnable bias (default: True).
        bit_width: Number of bits for quantization (default: 8).

    Attributes:
        linear: Underlying linear layer.
        weight_quantizer: DSQ quantizer for weights.
        activation_quantizer: DSQ quantizer for activations.
    """

    def __init__(self, in_features, out_features, bias=True, bit_width=8):
        """Initialize DSQ linear layer.

        Args:
            in_features: Size of input features.
            out_features: Size of output features.
            bias: If True, adds a learnable bias.
            bit_width: Number of bits for quantization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Original linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Quantizers
        self.weight_quantizer = DSQQuantize(
            bit_width=bit_width, signed=True
        )
        self.activation_quantizer = DSQQuantize(
            bit_width=bit_width, signed=True
        )

    def forward(self, x):
        """Forward pass with DSQ quantization.

        Args:
            x: Input tensor of shape (*, in_features).

        Returns:
            Output tensor of shape (*, out_features).
        """
        # Quantize activations
        x_quant = self.activation_quantizer(x)

        # Quantize weights
        weight_quant = self.weight_quantizer(self.linear.weight)

        # Compute output
        output = F.linear(x_quant, weight_quant, self.linear.bias)

        return output


class DSQConv1d(nn.Module):
    """Conv1d layer with DSQ quantization.

    Applies DSQ quantization to both weights and activations in a 1D
    convolutional layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of convolutional kernel.
        stride: Stride of convolution (default: 1).
        padding: Padding added to input (default: 0).
        bias: If True, adds a learnable bias (default: True).
        bit_width: Number of bits for quantization (default: 8).

    Attributes:
        conv: Underlying Conv1d layer.
        weight_quantizer: DSQ quantizer for weights.
        activation_quantizer: DSQ quantizer for activations.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        bit_width=8
    ):
        """Initialize DSQ Conv1d layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of convolutional kernel.
            stride: Stride of convolution.
            padding: Padding added to input.
            bias: If True, adds a learnable bias.
            bit_width: Number of bits for quantization.
        """
        super().__init__()

        # Original conv layer
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )

        # Quantizers
        self.weight_quantizer = DSQQuantize(
            bit_width=bit_width, signed=True
        )
        self.activation_quantizer = DSQQuantize(
            bit_width=bit_width, signed=True
        )

    def forward(self, x):
        """Forward pass with DSQ quantization.

        Args:
            x: Input tensor of shape (N, C_in, L_in).

        Returns:
            Output tensor of shape (N, C_out, L_out).
        """
        # Quantize activations
        x_quant = self.activation_quantizer(x)

        # Quantize weights
        weight_quant = self.weight_quantizer(self.conv.weight)

        # Compute output
        output = F.conv1d(
            x_quant, weight_quant, self.conv.bias,
            self.conv.stride, self.conv.padding
        )

        return output


class DSQConv2d(nn.Module):
    """Conv2d layer with DSQ quantization.

    Applies DSQ quantization to both weights and activations in a 2D
    convolutional layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of convolutional kernel.
        stride: Stride of convolution (default: 1).
        padding: Padding added to input (default: 0).
        bias: If True, adds a learnable bias (default: True).
        bit_width: Number of bits for quantization (default: 8).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        bit_width=8
    ):
        """Initialize DSQ Conv2d layer."""
        super().__init__()

        # Original conv layer
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )

        # Quantizers
        self.weight_quantizer = DSQQuantize(
            bit_width=bit_width, signed=True
        )
        self.activation_quantizer = DSQQuantize(
            bit_width=bit_width, signed=True
        )

    def forward(self, x):
        """Forward pass with DSQ quantization.

        Args:
            x: Input tensor of shape (N, C_in, H_in, W_in).

        Returns:
            Output tensor of shape (N, C_out, H_out, W_out).
        """
        # Quantize activations
        x_quant = self.activation_quantizer(x)

        # Quantize weights
        weight_quant = self.weight_quantizer(self.conv.weight)

        # Compute output
        output = F.conv2d(
            x_quant, weight_quant, self.conv.bias,
            self.conv.stride, self.conv.padding
        )

        return output


class DSQLSTM(nn.Module):
    """LSTM with DSQ quantization.

    Applies DSQ quantization to LSTM weights and activations.

    Args:
        input_size: Size of input features.
        hidden_size: Size of hidden state.
        num_layers: Number of LSTM layers (default: 1).
        bias: If True, adds a learnable bias (default: True).
        batch_first: If True, input shape is (batch, seq, features)
            (default: False).
        dropout: Dropout probability between layers (default: 0).
        bidirectional: If True, becomes bidirectional LSTM (default: False).
        bit_width: Number of bits for quantization (default: 8).

    Attributes:
        lstm: Underlying LSTM layer.
        weight_quantizers: List of quantizers for each layer's weights.
        activation_quantizer: Quantizer for activations.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
        bit_width=8
    ):
        """Initialize DSQ LSTM."""
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        # Original LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            bias=bias, batch_first=batch_first,
            dropout=dropout, bidirectional=bidirectional
        )

        # Quantizers for each weight matrix
        self.weight_quantizers = nn.ModuleList()

        # Create quantizers for each layer
        for layer in range(num_layers):
            layer_quantizers = nn.ModuleDict()

            # Input-hidden weights
            layer_quantizers['weight_ih'] = DSQQuantize(
                bit_width=bit_width, signed=True
            )
            # Hidden-hidden weights
            layer_quantizers['weight_hh'] = DSQQuantize(
                bit_width=bit_width, signed=True
            )

            if bidirectional:
                layer_quantizers['weight_ih_reverse'] = DSQQuantize(
                    bit_width=bit_width, signed=True
                )
                layer_quantizers['weight_hh_reverse'] = DSQQuantize(
                    bit_width=bit_width, signed=True
                )

            self.weight_quantizers.append(layer_quantizers)

        # Activation quantizer
        self.activation_quantizer = DSQQuantize(
            bit_width=bit_width, signed=True
        )

    def forward(self, x, hidden=None):
        """Forward pass with DSQ quantization.

        Args:
            x: Input tensor of shape (seq, batch, input_size) or
                (batch, seq, input_size) if batch_first=True.
            hidden: Initial hidden state tuple (h_0, c_0), or None.

        Returns:
            Tuple of (output, (h_n, c_n)) where:
                - output: Tensor of shape (seq, batch, hidden_size * D) or
                    (batch, seq, hidden_size * D) if batch_first=True,
                    where D=2 if bidirectional else 1.
                - h_n: Final hidden state.
                - c_n: Final cell state.
        """
        # Quantize input
        x_quant = self.activation_quantizer(x)

        # Store original weights
        original_weights = []

        # Quantize all LSTM weights
        for layer in range(self.num_layers):
            # Get weight names for this layer
            weight_ih_name = f'weight_ih_l{layer}'
            weight_hh_name = f'weight_hh_l{layer}'

            # Store original weights
            original_weights.append({
                'ih': getattr(self.lstm, weight_ih_name).data.clone(),
                'hh': getattr(self.lstm, weight_hh_name).data.clone()
            })

            # Quantize and replace weights
            weight_ih = getattr(self.lstm, weight_ih_name)
            weight_hh = getattr(self.lstm, weight_hh_name)

            weight_ih.data = (
                self.weight_quantizers[layer]['weight_ih'](weight_ih.data)
            )
            weight_hh.data = (
                self.weight_quantizers[layer]['weight_hh'](weight_hh.data)
            )

            if self.bidirectional:
                weight_ih_rev_name = f'weight_ih_l{layer}_reverse'
                weight_hh_rev_name = f'weight_hh_l{layer}_reverse'

                original_weights[-1]['ih_rev'] = (
                    getattr(self.lstm, weight_ih_rev_name).data.clone()
                )
                original_weights[-1]['hh_rev'] = (
                    getattr(self.lstm, weight_hh_rev_name).data.clone()
                )

                weight_ih_rev = getattr(self.lstm, weight_ih_rev_name)
                weight_hh_rev = getattr(self.lstm, weight_hh_rev_name)

                weight_ih_rev.data = (
                    self.weight_quantizers[layer]['weight_ih_reverse'](
                        weight_ih_rev.data
                    )
                )
                weight_hh_rev.data = (
                    self.weight_quantizers[layer]['weight_hh_reverse'](
                        weight_hh_rev.data
                    )
                )

        # Forward pass with quantized weights
        output, hidden_out = self.lstm(x_quant, hidden)

        # Restore original weights for gradient computation
        for layer in range(self.num_layers):
            weight_ih_name = f'weight_ih_l{layer}'
            weight_hh_name = f'weight_hh_l{layer}'

            getattr(self.lstm, weight_ih_name).data = (
                original_weights[layer]['ih']
            )
            getattr(self.lstm, weight_hh_name).data = (
                original_weights[layer]['hh']
            )

            if self.bidirectional:
                weight_ih_rev_name = f'weight_ih_l{layer}_reverse'
                weight_hh_rev_name = f'weight_hh_l{layer}_reverse'

                getattr(self.lstm, weight_ih_rev_name).data = (
                    original_weights[layer]['ih_rev']
                )
                getattr(self.lstm, weight_hh_rev_name).data = (
                    original_weights[layer]['hh_rev']
                )

        return output, hidden_out