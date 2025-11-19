"""AdaRound: Adaptive Rounding for Post-Training Quantization.

Based on "Up or Down? Adaptive Rounding for Post-Training Quantization"
(CVPR 2020).
Paper: https://arxiv.org/abs/2004.10568

This module implements AdaRound quantization with learnable rounding
parameters that adaptively determine whether to round up or down for each
weight value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaRoundQuantize(nn.Module):
    """AdaRound quantization with learnable rounding parameters.

    AdaRound learns whether to round each quantized value up or down using
    a learnable parameter alpha, which is optimized during post-training
    quantization or QAT.

    Args:
        bit_width: Number of bits for quantization (default: 8).
        signed: Use signed quantization range (default: True).

    Attributes:
        step_size: Learnable quantization step size.
        alpha: Learnable rounding parameter.
        temperature: Temperature for sigmoid function.
        initialized: Buffer tracking initialization status.
    """

    def __init__(self, bit_width=8, signed=True):
        """Initialize AdaRound quantizer.

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

        # Learnable step size
        self.step_size = nn.Parameter(torch.tensor(0.1))

        # Learnable rounding parameter (alpha)
        # Alpha determines whether to round up or down
        # Initialized to 0 (round to nearest)
        self.register_buffer('alpha_init', torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.zeros(1))

        # Temperature for sigmoid
        self.register_buffer('temperature', torch.tensor(1.0))

        # For initialization
        self.register_buffer('initialized', torch.tensor(0))

    def initialize_step_size(self, x):
        """Initialize step size based on input tensor statistics.

        Uses max-based initialization to ensure quantization range covers
        all values in the input.

        Args:
            x: Input tensor for initialization.
        """
        if self.initialized.item() == 0:
            with torch.no_grad():
                # Use max-based initialization to cover full range
                # This ensures quantization range covers all values
                max_val = x.detach().abs().max()

                # Initialize step size to cover full data range
                self.step_size.data = max_val / self.qmax

                self.initialized.fill_(1)

    def forward(self, x):
        """Forward pass with AdaRound quantization.

        During training, uses adaptive rounding with learnable alpha.
        During inference, uses standard rounding.

        Args:
            x: Input tensor to quantize.

        Returns:
            Quantized tensor (with STE during training).
        """
        if self.training:
            # Initialize on first forward pass
            self.initialize_step_size(x)

            # Adaptive rounding during training
            # h(alpha) = clip(sigmoid((alpha-0.5)/temperature), 0, 1)
            h_alpha = torch.sigmoid((self.alpha - 0.5) / self.temperature)
            h_alpha = torch.clamp(h_alpha, 0, 1)

            # Compute floor and ceil
            x_scaled = x / self.step_size
            x_floor = torch.floor(x_scaled)

            # Adaptive rounding: x_round = x_floor + h(alpha)
            x_rounded = x_floor + h_alpha

            # Clamp to quantization range
            x_rounded = torch.clamp(x_rounded, self.qmin, self.qmax)

            # Dequantize
            x_quant = x_rounded * self.step_size

            # Straight-through estimator for gradients
            return x + (x_quant - x).detach()
        else:
            # Standard rounding during inference
            x_scaled = x / self.step_size
            x_rounded = torch.round(x_scaled)
            x_rounded = torch.clamp(x_rounded, self.qmin, self.qmax)
            x_quant = x_rounded * self.step_size
            return x_quant


class AdaRoundLinear(nn.Module):
    """Linear layer with AdaRound quantization.

    Applies AdaRound quantization to both weights and activations.

    Args:
        in_features: Size of input features.
        out_features: Size of output features.
        bias: If True, adds a learnable bias (default: True).
        bit_width: Number of bits for quantization (default: 8).
    """

    def __init__(self, in_features, out_features, bias=True, bit_width=8):
        """Initialize AdaRound linear layer."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Original linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Quantizers
        self.weight_quantizer = AdaRoundQuantize(
            bit_width=bit_width, signed=True
        )
        self.activation_quantizer = AdaRoundQuantize(
            bit_width=bit_width, signed=True
        )

    def forward(self, x):
        """Forward pass with AdaRound quantization.

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


class AdaRoundConv1d(nn.Module):
    """Conv1d layer with AdaRound quantization.

    Applies AdaRound quantization to both weights and activations.

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
        """Initialize AdaRound Conv1d layer."""
        super().__init__()

        # Original conv layer
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )

        # Quantizers
        self.weight_quantizer = AdaRoundQuantize(
            bit_width=bit_width, signed=True
        )
        self.activation_quantizer = AdaRoundQuantize(
            bit_width=bit_width, signed=True
        )

    def forward(self, x):
        """Forward pass with AdaRound quantization.

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


class AdaRoundConv2d(nn.Module):
    """Conv2d layer with AdaRound quantization.

    Applies AdaRound quantization to both weights and activations.

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
        """Initialize AdaRound Conv2d layer."""
        super().__init__()

        # Original conv layer
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )

        # Quantizers
        self.weight_quantizer = AdaRoundQuantize(
            bit_width=bit_width, signed=True
        )
        self.activation_quantizer = AdaRoundQuantize(
            bit_width=bit_width, signed=True
        )

    def forward(self, x):
        """Forward pass with AdaRound quantization.

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


class AdaRoundWeightQuantizer(nn.Module):
    """
    Per-channel AdaRound quantizer for convolution weights.

    Closely follows the EfficientDL reference implementation: every output
    channel gets its own rounding parameter that is encouraged to converge
    to {0, 1}, which stabilizes training compared to a single global alpha.
    """

    def __init__(self, bits=8, per_channel=True, channel_axis=0):
        super().__init__()
        assert bits >= 2, "AdaRound requires at least 2 bits"
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
        self.per_channel = per_channel
        self.channel_axis = channel_axis

        self.register_buffer('scale_initialized', torch.tensor(False))
        self.register_buffer('alpha_initialized', torch.tensor(False))

        self.scale = None
        self.alpha = None
        self.hard_round_in_eval = True

    def _broadcast(self, weight, value):
        if not self.per_channel:
            return value
        view = [1] * weight.dim()
        view[self.channel_axis] = -1
        return value.view(view)

    @torch.no_grad()
    def _init_scale(self, weight):
        if self.per_channel:
            permuted = weight.transpose(0, self.channel_axis).contiguous().flatten(1)
            scale = (permuted.abs().max(dim=1).values / max(self.qmax, 1)).clamp(min=1e-8)
        else:
            scale = (weight.abs().max() / max(self.qmax, 1)).clamp(min=1e-8)

        self.scale = scale.detach()
        self.scale_initialized.fill_(True)

    @torch.no_grad()
    def _init_alpha(self, weight):
        scale = self._broadcast(weight, self.scale.to(weight.device, weight.dtype))
        y = (weight / scale).detach()
        fractional = (y - torch.floor(y)).clamp(0, 1 - 1e-6)
        alpha = torch.log(fractional / (1 - fractional + 1e-6) + 1e-12)
        self.alpha = nn.Parameter(alpha.to(dtype=weight.dtype, device=weight.device))
        self.alpha_initialized.fill_(True)

    def forward(self, weight):
        if not self.scale_initialized:
            self._init_scale(weight)
        if (self.alpha is None) or (not self.alpha_initialized):
            self._init_alpha(weight)

        scale = self._broadcast(weight, self.scale.to(weight.device, weight.dtype))
        y = weight / scale
        base = torch.floor(y)

        r_soft = torch.sigmoid(self.alpha)
        if (not self.training) and self.hard_round_in_eval:
            r = (r_soft >= 0.5).to(weight.dtype)
        else:
            r = r_soft

        quant = torch.clamp(base + r, self.qmin, self.qmax)
        w_q = scale * quant

        return weight + (w_q - weight).detach()

    def regularization(self, lam=1e-4):
        if self.alpha is None:
            device = self.scale.device if isinstance(self.scale, torch.Tensor) else 'cpu'
            return torch.tensor(0.0, device=device)
        r = torch.sigmoid(self.alpha)
        reg = (1.0 - (2.0 * r - 1.0).abs()).mean()
        return lam * reg

    def set_hard_round(self, enabled=True):
        self.hard_round_in_eval = enabled


class AdaRoundLSTM(nn.Module):
    """LSTM with AdaRound quantization"""

    def __init__(self, input_size, hidden_size, num_layers=1,
                 bias=True, batch_first=False, dropout=0,
                 bidirectional=False, bit_width=8):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        # Original LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           bias=bias, batch_first=batch_first,
                           dropout=dropout, bidirectional=bidirectional)

        # Quantizers for each weight matrix
        self.weight_quantizers = nn.ModuleList()

        # Create quantizers for each layer
        for layer in range(num_layers):
            layer_quantizers = nn.ModuleDict()

            # Input-hidden weights
            layer_quantizers['weight_ih'] = AdaRoundQuantize(bit_width=bit_width, signed=True)
            # Hidden-hidden weights
            layer_quantizers['weight_hh'] = AdaRoundQuantize(bit_width=bit_width, signed=True)

            if bidirectional:
                layer_quantizers['weight_ih_reverse'] = AdaRoundQuantize(bit_width=bit_width, signed=True)
                layer_quantizers['weight_hh_reverse'] = AdaRoundQuantize(bit_width=bit_width, signed=True)

            self.weight_quantizers.append(layer_quantizers)

        # Activation quantizer
        self.activation_quantizer = AdaRoundQuantize(bit_width=bit_width, signed=True)

    def forward(self, x, hidden=None):
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

            weight_ih.data = self.weight_quantizers[layer]['weight_ih'](weight_ih.data)
            weight_hh.data = self.weight_quantizers[layer]['weight_hh'](weight_hh.data)

            if self.bidirectional:
                weight_ih_rev_name = f'weight_ih_l{layer}_reverse'
                weight_hh_rev_name = f'weight_hh_l{layer}_reverse'

                original_weights[-1]['ih_rev'] = getattr(self.lstm, weight_ih_rev_name).data.clone()
                original_weights[-1]['hh_rev'] = getattr(self.lstm, weight_hh_rev_name).data.clone()

                weight_ih_rev = getattr(self.lstm, weight_ih_rev_name)
                weight_hh_rev = getattr(self.lstm, weight_hh_rev_name)

                weight_ih_rev.data = self.weight_quantizers[layer]['weight_ih_reverse'](weight_ih_rev.data)
                weight_hh_rev.data = self.weight_quantizers[layer]['weight_hh_reverse'](weight_hh_rev.data)

        # Forward pass with quantized weights
        output, hidden_out = self.lstm(x_quant, hidden)

        # Restore original weights for gradient computation
        for layer in range(self.num_layers):
            weight_ih_name = f'weight_ih_l{layer}'
            weight_hh_name = f'weight_hh_l{layer}'

            getattr(self.lstm, weight_ih_name).data = original_weights[layer]['ih']
            getattr(self.lstm, weight_hh_name).data = original_weights[layer]['hh']

            if self.bidirectional:
                weight_ih_rev_name = f'weight_ih_l{layer}_reverse'
                weight_hh_rev_name = f'weight_hh_l{layer}_reverse'

                getattr(self.lstm, weight_ih_rev_name).data = original_weights[layer]['ih_rev']
                getattr(self.lstm, weight_hh_rev_name).data = original_weights[layer]['hh_rev']

        return output, hidden_out
