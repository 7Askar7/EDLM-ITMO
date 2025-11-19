"""LSTM-based text classifier for sentiment analysis.

This module implements LSTM-based text classifiers for binary sentiment
classification tasks using the IMDB movie reviews dataset.

Dataset: IMDB movie reviews
Metric: ROC-AUC
"""

import sys
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    """LSTM classifier for text classification.

    Architecture consists of:
    - Embedding layer for word representations
    - Bidirectional LSTM for sequence encoding
    - Dropout for regularization
    - Fully connected layers with batch normalization
    - Binary classification output

    Args:
        vocab_size: Size of vocabulary (default: 20000).
        embedding_dim: Dimension of word embeddings (default: 128).
        hidden_dim: Dimension of LSTM hidden state (default: 256).
        num_layers: Number of LSTM layers (default: 2).
        dropout: Dropout probability (default: 0.5).
        bidirectional: Whether to use bidirectional LSTM (default: True).
        num_classes: Number of output classes (default: 1).
    """

    def __init__(
        self,
        vocab_size: int = 20000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        num_classes: int = 1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers
        fc_input_dim = hidden_dim * self.num_directions
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    if param.dim() >= 2:
                        nn.init.orthogonal_(param)
                elif 'embedding' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
                else:
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, lengths=None):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len)
            lengths: Actual lengths of sequences (optional)

        Returns:
            output: Predictions of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = self.dropout(embedded)

        # Pack padded sequence if lengths are provided
        if lengths is not None:
            # Sort by length (required for pack_padded_sequence)
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            embedded_sorted = embedded[sorted_idx]

            packed = nn.utils.rnn.pack_padded_sequence(
                embedded_sorted,
                lengths_sorted.cpu(),
                batch_first=True,
                enforce_sorted=True
            )

            # LSTM
            packed_output, (hidden, cell) = self.lstm(packed)

            # Unpack
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

            # Unsort
            _, unsorted_idx = sorted_idx.sort()
            output = output[unsorted_idx]
            hidden = hidden[:, unsorted_idx, :]
        else:
            # LSTM without packing
            output, (hidden, cell) = self.lstm(embedded)

        # Use last hidden state from both directions
        if self.bidirectional:
            # Concatenate last hidden states from forward and backward
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        # Fully connected layers
        x = self.dropout(hidden)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x

    def predict_proba(self, x, lengths=None):
        """Get prediction probabilities using sigmoid activation.

        Args:
            x: Input tensor of shape (batch_size, seq_len).
            lengths: Optional tensor of actual sequence lengths.

        Returns:
            Tensor of shape (batch_size, num_classes) with probabilities
            in range [0, 1].
        """
        logits = self.forward(x, lengths)
        return torch.sigmoid(logits)


class QuantizedLSTMClassifier(nn.Module):
    """LSTM classifier with quantization-aware training.

    This variant uses quantized linear layers for the fully connected
    layers to enable efficient deployment on resource-constrained devices.
    The LSTM and embedding layers remain in full precision.

    Args:
        vocab_size: Size of vocabulary (default: 20000).
        embedding_dim: Dimension of word embeddings (default: 128).
        hidden_dim: Dimension of LSTM hidden state (default: 256).
        num_layers: Number of LSTM layers (default: 2).
        dropout: Dropout probability (default: 0.5).
        bidirectional: Whether to use bidirectional LSTM (default: True).
        num_classes: Number of output classes (default: 1).
        quantizer_type: Type of quantizer to use (default: 'lsq').
        bit_width: Bit width for quantization (default: 8).
    """

    def __init__(
        self,
        vocab_size: int = 20000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        num_classes: int = 1,
        quantizer_type: str = 'lsq',
        bit_width: int = 8
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Import quantized layers (late import to avoid circular dependencies)
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from quantization.fake_quantize import QuantizedLinear  # noqa: E402

        # Embedding layer (not quantized)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layer (not quantized)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Quantized fully connected layers
        fc_input_dim = hidden_dim * self.num_directions
        self.fc1 = QuantizedLinear(fc_input_dim, 128,
                                    quantizer_type=quantizer_type,
                                    bit_width=bit_width)
        self.fc2 = QuantizedLinear(128, 64,
                                    quantizer_type=quantizer_type,
                                    bit_width=bit_width)
        self.fc3 = QuantizedLinear(64, num_classes,
                                    quantizer_type=quantizer_type,
                                    bit_width=bit_width)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    if param.dim() >= 2:
                        nn.init.orthogonal_(param)
                elif 'embedding' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
                else:
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, lengths=None):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len)
            lengths: Actual lengths of sequences (optional)

        Returns:
            output: Predictions of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = self.dropout(embedded)

        # Pack padded sequence if lengths are provided
        if lengths is not None:
            # Sort by length (required for pack_padded_sequence)
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            embedded_sorted = embedded[sorted_idx]

            packed = nn.utils.rnn.pack_padded_sequence(
                embedded_sorted,
                lengths_sorted.cpu(),
                batch_first=True,
                enforce_sorted=True
            )

            # LSTM
            packed_output, (hidden, cell) = self.lstm(packed)

            # Unpack
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

            # Unsort
            _, unsorted_idx = sorted_idx.sort()
            output = output[unsorted_idx]
            hidden = hidden[:, unsorted_idx, :]
        else:
            # LSTM without packing
            output, (hidden, cell) = self.lstm(embedded)

        # Use last hidden state from both directions
        if self.bidirectional:
            # Concatenate last hidden states from forward and backward
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        # Fully connected layers
        x = self.dropout(hidden)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x

    def predict_proba(self, x, lengths=None):
        """Get prediction probabilities using sigmoid activation.

        Args:
            x: Input tensor of shape (batch_size, seq_len).
            lengths: Optional tensor of actual sequence lengths.

        Returns:
            Tensor of shape (batch_size, num_classes) with probabilities
            in range [0, 1].
        """
        logits = self.forward(x, lengths)
        return torch.sigmoid(logits)
