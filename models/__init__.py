"""Models package for neural network architectures.

This package contains implementations of various deep learning models:
- LSTM Classifier: Text classification with bidirectional LSTM
- ESPCN: Efficient Sub-Pixel CNN for super-resolution
- SASRec: Self-Attentive Sequential Recommendation model

Each model is available in both standard and quantized variants for
efficient deployment.
"""

from .espcn import ESPCN
from .lstm_classifier import LSTMClassifier, QuantizedLSTMClassifier
from .sasrec import SASRec

__all__ = [
    'LSTMClassifier',
    'QuantizedLSTMClassifier',
    'ESPCN',
    'SASRec'
]
