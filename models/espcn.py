"""ESPCN (Efficient Sub-Pixel CNN) for Super Resolution.

This module implements the Efficient Sub-Pixel Convolutional Neural Network
for real-time single image and video super-resolution.

Reference:
    "Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network" (CVPR 2016)

Dataset: BSD100, Set5, Set14
Metric: PSNR (Peak Signal-to-Noise Ratio)
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as nnq

from quantization.adaround import AdaRoundWeightQuantizer
from quantization.apot import APoTQuantize
from quantization.dsq import DSQQuantize
from quantization.lsq import LSQQuantizer
from quantization.pact import PACTQuantizerV2


class BaseESPCN(nn.Module):
    """ESPCN backbone with overridable quantization hooks.

    This base class provides the core ESPCN architecture with hooks for
    weight and activation quantization that can be overridden in subclasses.

    Args:
        upscale_factor: Image upscaling factor (default: 3).
        num_channels: Number of input/output channels (default: 1).
        feature_channels: Number of feature channels (default: 64).
    """

    def __init__(
        self,
        upscale_factor: int = 3,
        num_channels: int = 1,
        feature_channels: int = 64
    ):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.num_channels = num_channels
        self.feature_channels = feature_channels

        # Feature extraction layers
        self.conv1 = nn.Conv2d(num_channels, feature_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature_channels // 2, feature_channels // 4, kernel_size=3, padding=1)

        # Sub-pixel convolution layer
        self.conv4 = nn.Conv2d(
            feature_channels // 4,
            num_channels * (upscale_factor ** 2),
            kernel_size=3,
            padding=1
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._init_weights()

    def _init_weights(self):
        """Initialize convolutional layer weights.

        Uses Kaiming initialization for feature extraction layers and
        normal initialization for the sub-pixel convolution layer.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module is self.conv4:
                    nn.init.normal_(module.weight, mean=0.0, std=0.001)
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    # Quantization hooks ----------------------------------------------------
    def quantize_weight(
        self,
        layer_name: str,
        weight: torch.Tensor
    ) -> torch.Tensor:
        """Hook for weight quantization (override in subclasses).

        Args:
            layer_name: Name of the layer being quantized.
            weight: Weight tensor to quantize.

        Returns:
            Quantized weight tensor (or original if not overridden).
        """
        return weight

    def quantize_activation(
        self,
        layer_name: str,
        activation: torch.Tensor
    ) -> torch.Tensor:
        """Hook for activation quantization (override in subclasses).

        Args:
            layer_name: Name of the layer producing the activation.
            activation: Activation tensor to quantize.

        Returns:
            Quantized activation tensor (or original if not overridden).
        """
        return activation

    def _conv_forward(
        self,
        layer: nn.Conv2d,
        x: torch.Tensor,
        layer_name: str
    ) -> torch.Tensor:
        """Forward pass for a convolutional layer with weight quantization.

        Args:
            layer: Convolutional layer to apply.
            x: Input tensor.
            layer_name: Name of the layer for quantization hook.

        Returns:
            Output tensor after convolution.
        """
        if isinstance(layer, nnq.Conv2d):
            return layer(x)

        weight = self.quantize_weight(layer_name, layer.weight)
        return F.conv2d(
            x,
            weight,
            layer.bias,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )

    def forward(self, x):
        """Forward pass through the ESPCN network.

        Args:
            x: Input low-resolution image tensor of shape
                (batch, num_channels, height, width).

        Returns:
            Super-resolved image tensor of shape
            (batch, num_channels, height*upscale_factor,
            width*upscale_factor).
        """
        x = self._conv_forward(self.conv1, x, 'conv1')
        x = F.relu(x)
        x = self.quantize_activation('conv1', x)

        x = self._conv_forward(self.conv2, x, 'conv2')
        x = F.relu(x)
        x = self.quantize_activation('conv2', x)

        x = self._conv_forward(self.conv3, x, 'conv3')
        x = F.relu(x)
        x = self.quantize_activation('conv3', x)

        x = self._conv_forward(self.conv4, x, 'conv4')
        x = self.pixel_shuffle(x)
        return x


class ESPCN(BaseESPCN):
    """Floating-point ESPCN model without quantization.

    This is the standard ESPCN implementation without any quantization.
    Use this for training and inference when model size is not a constraint.

    Args:
        upscale_factor: Image upscaling factor (default: 3).
        num_channels: Number of input/output channels (default: 1).
        feature_channels: Number of feature channels (default: 64).
    """

    def __init__(
        self,
        upscale_factor: int = 3,
        num_channels: int = 1,
        feature_channels: int = 64
    ):
        super().__init__(
            upscale_factor=upscale_factor,
            num_channels=num_channels,
            feature_channels=feature_channels
        )


class ESPCNDeep(nn.Module):
    """Deeper ESPCN variant with more feature extraction layers.

    This variant adds more convolutional layers for improved quality
    at the cost of increased computational complexity.

    Args:
        upscale_factor: Image upscaling factor (default: 3).
        num_channels: Number of input/output channels (default: 1).
        feature_channels: Number of feature channels (default: 64).
        num_layers: Number of feature extraction layers (default: 5).
    """

    def __init__(
        self,
        upscale_factor: int = 3,
        num_channels: int = 1,
        feature_channels: int = 64,
        num_layers: int = 5
    ):
        super().__init__()

        self.upscale_factor = upscale_factor
        self.num_channels = num_channels
        self.feature_channels = feature_channels
        self.num_layers = num_layers

        # First conv layer
        self.conv_first = nn.Conv2d(num_channels, feature_channels, kernel_size=5, padding=2)

        # Feature extraction layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            in_ch = feature_channels if i == 0 else feature_channels // 2
            out_ch = feature_channels // 2
            self.conv_layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            )

        # Sub-pixel convolution
        self.conv_last = nn.Conv2d(
            feature_channels // 2,
            num_channels * (upscale_factor ** 2),
            kernel_size=3,
            padding=1
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize convolutional layer weights.

        Uses Kaiming initialization for feature extraction layers and
        normal initialization for the sub-pixel convolution layer.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m == self.conv_last:
                    nn.init.normal_(m.weight, mean=0.0, std=0.001)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the deep ESPCN network.

        Args:
            x: Input low-resolution image tensor of shape
                (batch, num_channels, height, width).

        Returns:
            Super-resolved image tensor of shape
            (batch, num_channels, height*upscale_factor,
            width*upscale_factor).
        """
        x = F.relu(self.conv_first(x))

        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))

        x = self.conv_last(x)
        x = self.pixel_shuffle(x)

        return x


class QuantizedESPCN(BaseESPCN):
    """ESPCN with modular quantization-aware training.

    This variant implements quantization-aware training where each
    convolution weight tensor is quantized independently and activations
    are quantized after ReLU activation (conv1-3).

    Args:
        upscale_factor: Image upscaling factor (default: 3).
        num_channels: Number of input/output channels (default: 1).
        feature_channels: Number of feature channels (default: 64).
        quantizer_type: Type of quantizer ('lsq', 'pact', 'adaround',
            'apot', 'dsq') (default: 'lsq').
        bit_width: Bit width for weight quantization (default: 8).
        activation_bit_width: Bit width for activation quantization.
            If None, uses same as weight bit_width (default: None).
    """

    def __init__(
        self,
        upscale_factor: int = 3,
        num_channels: int = 1,
        feature_channels: int = 64,
        quantizer_type: str = 'lsq',
        bit_width: int = 8,
        activation_bit_width: Optional[int] = None,
    ):
        super().__init__(
            upscale_factor=upscale_factor,
            num_channels=num_channels,
            feature_channels=feature_channels
        )
        self.quantizer_type = quantizer_type
        self.bit_width = bit_width
        self.activation_bit_width = activation_bit_width or bit_width

        self.weight_enabled = True
        self.activation_enabled = True

        self.weight_quantizers = nn.ModuleDict(
            self._build_weight_quantizers()
        )
        self.activation_quantizers = nn.ModuleDict(
            self._build_activation_quantizers()
        )

    # Helper methods --------------------------------------------------------
    def _build_weight_quantizers(self) -> Dict[str, nn.Module]:
        """Build weight quantizers for each convolutional layer.

        Returns:
            Dictionary mapping layer names to quantizer modules.
        """
        quantizers: Dict[str, nn.Module] = {}
        for layer_name in ('conv1', 'conv2', 'conv3', 'conv4'):
            quantizer = self._create_weight_quantizer(layer_name)
            if quantizer is not None:
                quantizers[layer_name] = quantizer
        return quantizers

    def _create_weight_quantizer(
        self,
        layer_name: str
    ) -> Optional[nn.Module]:
        """Create a weight quantizer for a specific layer.

        Args:
            layer_name: Name of the layer to create quantizer for.

        Returns:
            Quantizer module or None.

        Raises:
            ValueError: If quantizer_type is unsupported.
        """
        if self.quantizer_type in ('lsq', 'pact'):
            return LSQQuantizer(self.bit_width, symmetric=True)
        if self.quantizer_type == 'adaround':
            return AdaRoundWeightQuantizer(
                bits=self.bit_width,
                per_channel=True,
                channel_axis=0
            )
        if self.quantizer_type == 'apot':
            return APoTQuantize(
                num_bits=self.bit_width,
                weight_mode=True
            )
        if self.quantizer_type == 'dsq':
            return DSQQuantize(bit_width=self.bit_width, signed=True)
        raise ValueError(
            f"Unsupported quantization type: {self.quantizer_type}"
        )

    def _build_activation_quantizers(self) -> Dict[str, nn.Module]:
        """Build activation quantizers for conv1-3 layers.

        Returns:
            Dictionary mapping layer names to quantizer modules.
        """
        channels = {
            'conv1': self.feature_channels,
            'conv2': self.feature_channels // 2,
            'conv3': self.feature_channels // 4,
        }
        quantizers: Dict[str, nn.Module] = {}
        for name, num_channels in channels.items():
            quantizer = self._create_activation_quantizer(name, num_channels)
            if quantizer is not None:
                quantizers[name] = quantizer
        return quantizers

    def _create_activation_quantizer(
        self,
        layer_name: str,
        num_channels: int
    ) -> Optional[nn.Module]:
        """Create an activation quantizer for a specific layer.

        Args:
            layer_name: Name of the layer to create quantizer for.
            num_channels: Number of channels in the activation.

        Returns:
            Quantizer module or None.

        Raises:
            ValueError: If quantizer_type is unsupported.
        """
        if self.quantizer_type == 'lsq':
            return LSQQuantizer(
                self.activation_bit_width,
                symmetric=True,
                all_positive=True
            )
        if self.quantizer_type == 'pact':
            return PACTQuantizerV2(
                bit_width=self.activation_bit_width,
                symmetric=False,
                per_channel=True,
                num_channels=num_channels,
            )
        if self.quantizer_type == 'adaround':
            return None  # weight-only quantization
        if self.quantizer_type == 'apot':
            return APoTQuantize(
                num_bits=self.activation_bit_width,
                weight_mode=False
            )
        if self.quantizer_type == 'dsq':
            return DSQQuantize(
                bit_width=self.activation_bit_width,
                signed=False
            )
        raise ValueError(
            f"Unsupported quantization type: {self.quantizer_type}"
        )

    # Quantization hooks ----------------------------------------------------
    def quantize_weight(
        self,
        layer_name: str,
        weight: torch.Tensor
    ) -> torch.Tensor:
        """Apply weight quantization if enabled.

        Args:
            layer_name: Name of the layer being quantized.
            weight: Weight tensor to quantize.

        Returns:
            Quantized weight tensor.
        """
        if not self.weight_enabled or layer_name not in self.weight_quantizers:
            return weight
        return self.weight_quantizers[layer_name](weight)

    def quantize_activation(
        self,
        layer_name: str,
        activation: torch.Tensor
    ) -> torch.Tensor:
        """Apply activation quantization if enabled.

        Args:
            layer_name: Name of the layer producing the activation.
            activation: Activation tensor to quantize.

        Returns:
            Quantized activation tensor.
        """
        if not self.activation_enabled or layer_name not in self.activation_quantizers:
            return activation
        return self.activation_quantizers[layer_name](activation)

    # Control methods -------------------------------------------------------
    def set_weight_quantization(self, enabled: bool):
        """Enable or disable weight quantization.

        Args:
            enabled: Whether to enable weight quantization.
        """
        self.weight_enabled = enabled

    def set_activation_quantization(self, enabled: bool):
        """Enable or disable activation quantization.

        Args:
            enabled: Whether to enable activation quantization.
        """
        self.activation_enabled = enabled

    def regularization_loss(self, coefficient: float = 0.0) -> torch.Tensor:
        """Compute regularization loss from quantizers.

        Args:
            coefficient: Regularization coefficient (default: 0.0).

        Returns:
            Total regularization loss as a scalar tensor.
        """
        if coefficient <= 0:
            return torch.tensor(0.0, device=self.conv1.weight.device)

        reg_terms = []
        for quantizer in self.weight_quantizers.values():
            if hasattr(quantizer, 'regularization'):
                reg_terms.append(quantizer.regularization(coefficient))

        if not reg_terms:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return torch.stack(reg_terms).sum()


def calculate_psnr(img1, img2, max_value=1.0):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        img1: First image tensor.
        img2: Second image tensor (same shape as img1).
        max_value: Maximum pixel value, 1.0 for normalized images
            (default: 1.0).

    Returns:
        PSNR value in dB. Returns infinity if images are identical.
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """Calculate SSIM (Structural Similarity Index) between two images.

    This is a simplified SSIM implementation. For production use,
    consider using pytorch-ssim or other dedicated libraries.

    Args:
        img1: First image tensor.
        img2: Second image tensor (same shape as img1).
        window_size: Size of Gaussian window for local statistics
            (default: 11).
        size_average: Whether to average SSIM across all channels
            (default: True).

    Returns:
        SSIM value between 0 and 1. Returns mean value if size_average
        is True, otherwise returns per-channel values.
    """
    # Simplified SSIM calculation using average pooling
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.avg_pool2d(
            img1 * img1, window_size, stride=1, padding=window_size // 2
        ) - mu1_sq
    )
    sigma2_sq = (
        F.avg_pool2d(
            img2 * img2, window_size, stride=1, padding=window_size // 2
        ) - mu2_sq
    )
    sigma12 = (
        F.avg_pool2d(
            img1 * img2, window_size, stride=1, padding=window_size // 2
        ) - mu1_mu2
    )

    ssim_map = (
        (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    ) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
