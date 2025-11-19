"""Quantization methods package.

This package provides various quantization methods for neural networks:
- LSQ: Learned Step Size Quantization
- PACT: Parameterized Clipping Activation
- AdaRound: Adaptive Rounding
- APoT: Additive Powers-of-Two Quantization
- DSQ: Differentiable Soft Quantization
- FakeQuantize: Wrapper for fake quantization
"""

from .adaround import (
    AdaRoundConv1d,
    AdaRoundConv2d,
    AdaRoundLinear,
    AdaRoundLSTM,
    AdaRoundQuantize,
    AdaRoundWeightQuantizer,
)
from .apot import (
    APoTConv1d,
    APoTConv2d,
    APoTLinear,
    APoTQuantize,
    generate_apot_codebook,
)
from .dsq import DSQConv1d, DSQConv2d, DSQLinear, DSQLSTM, DSQQuantize
from .fake_quantize import (
    FakeQuantize,
    QuantizedConv1d,
    QuantizedConv2d,
    QuantizedLinear,
    QuantizedLSTMCell,
)
from .lsq import LSQPlusQuantizer, LSQQuantizer
from .pact import (
    PACTActivation,
    PACTQuantizer,
    PACTQuantizerV2,
)

__all__ = [
    # LSQ
    'LSQQuantizer',
    'LSQPlusQuantizer',
    # PACT
    'PACTQuantizer',
    'PACTQuantizerV2',
    'PACTActivation',
    # AdaRound
    'AdaRoundQuantize',
    'AdaRoundWeightQuantizer',
    'AdaRoundLinear',
    'AdaRoundConv1d',
    'AdaRoundConv2d',
    'AdaRoundLSTM',
    # APoT
    'APoTQuantize',
    'APoTLinear',
    'APoTConv1d',
    'APoTConv2d',
    'generate_apot_codebook',
    # DSQ
    'DSQQuantize',
    'DSQLinear',
    'DSQConv1d',
    'DSQConv2d',
    'DSQLSTM',
    # Fake Quantize
    'FakeQuantize',
    'QuantizedLinear',
    'QuantizedConv1d',
    'QuantizedConv2d',
    'QuantizedLSTMCell',
]
