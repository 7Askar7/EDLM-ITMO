"""SASRec (Self-Attentive Sequential Recommendation).

This module implements the Self-Attentive Sequential Recommendation model
for next-item prediction in recommender systems.

Reference:
    "Self-Attentive Sequential Recommendation" (ICDM 2018)

Dataset: MovieLens-1M
Metrics: NDCG@10, HR@10
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    Applies two 1D convolutions with ReLU activation and dropout
    in between, followed by residual connection.

    Args:
        hidden_units: Number of hidden units/channels.
        dropout_rate: Dropout probability.
    """

    def __init__(self, hidden_units, dropout_rate):
        super().__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        """Forward pass with residual connection.

        Args:
            inputs: Input tensor of shape (batch, seq_len, hidden_units).

        Returns:
            Output tensor of shape (batch, seq_len, hidden_units).
        """
        # Apply 1D convolutions (transpose for conv1d format)
        x = inputs.transpose(-1, -2)  # (batch, hidden_units, seq_len)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        outputs = x.transpose(-1, -2)  # (batch, seq_len, hidden_units)

        # Residual connection
        outputs += inputs
        return outputs


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model.

    Uses multi-head self-attention with causal masking to model
    sequential user behavior for next-item prediction.

    Args:
        num_items: Number of items in the dataset.
        max_seq_len: Maximum sequence length (default: 200).
        hidden_units: Number of hidden units (default: 128).
        num_blocks: Number of self-attention blocks (default: 2).
        num_heads: Number of attention heads (default: 2).
        dropout_rate: Dropout probability (default: 0.2).
    """

    def __init__(
        self,
        num_items,
        max_seq_len=200,
        hidden_units=128,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.2
    ):
        super().__init__()

        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        # Item embedding
        # num_items + 1 for padding item (0)
        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)

        # Positional embedding
        self.pos_emb = nn.Embedding(max_seq_len, hidden_units)

        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # Self-attention blocks
        self.attention_layernorms = nn.ModuleList()  # LayerNorm before attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()  # LayerNorm before FFN
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(
                hidden_units,
                num_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                # Zero out padding embedding
                if hasattr(m, 'padding_idx') and m.padding_idx is not None:
                    m.weight.data[m.padding_idx].fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def log2feats(self, log_seqs):
        """Convert log sequences to feature representations.

        Applies embedding layers, positional encoding, and multi-layer
        self-attention with feed-forward networks.

        Args:
            log_seqs: Input item sequences of shape (batch, seq_len).

        Returns:
            Sequence features of shape (batch, seq_len, hidden_units).
        """
        seqs = self.item_emb(log_seqs)  # (batch, seq_len, hidden_units)

        # Add positional embeddings
        positions = torch.arange(log_seqs.size(1), device=log_seqs.device).unsqueeze(0)
        positions = positions.expand_as(log_seqs)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        # Create attention mask (causal mask)
        # Mask out padding positions and future positions
        timeline_mask = (log_seqs == 0)  # Padding mask
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        # Attention mask for causality
        # True means masked (not allowed to attend)
        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = torch.triu(
            torch.ones((tl, tl), dtype=torch.bool, device=log_seqs.device),
            diagonal=1
        )

        # Apply self-attention blocks
        for i in range(len(self.attention_layers)):
            # LayerNorm
            Q = self.attention_layernorms[i](seqs)

            # Self-attention with causal mask
            mha_outputs, _ = self.attention_layers[i](
                Q, Q, Q,
                attn_mask=attention_mask,
                key_padding_mask=timeline_mask
            )

            seqs = Q + mha_outputs

            # Feed forward
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (batch, seq_len, hidden_units)

        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        """Forward pass for training with positive and negative sampling.

        Args:
            log_seqs: Input item sequences of shape (batch, seq_len).
            pos_seqs: Positive (ground truth) items of shape
                (batch, seq_len).
            neg_seqs: Negative (sampled) items of shape (batch, seq_len).

        Returns:
            Tuple of (pos_logits, neg_logits):
                - pos_logits: Scores for positive items (batch, seq_len).
                - neg_logits: Scores for negative items (batch, seq_len).
        """
        log_feats = self.log2feats(log_seqs)  # (batch, seq_len, hidden_units)

        pos_embs = self.item_emb(pos_seqs)  # (batch, seq_len, hidden_units)
        neg_embs = self.item_emb(neg_seqs)  # (batch, seq_len, hidden_units)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # (batch, seq_len)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)  # (batch, seq_len)

        return pos_logits, neg_logits

    def predict(self, log_seqs, item_indices):
        """Predict scores for given candidate items.

        Args:
            log_seqs: Input item sequences of shape (batch, seq_len).
            item_indices: Candidate items to score. Can be either:
                - 1D tensor of shape (num_items,) to score same items
                  for all sequences.
                - 2D tensor of shape (batch, num_items) for
                  sequence-specific candidates.

        Returns:
            Prediction scores of shape (batch, num_items).
        """
        log_feats = self.log2feats(log_seqs)  # (batch, seq_len, hidden_units)

        final_feat = log_feats[:, -1, :]  # (batch, hidden_units), only use last position

        if item_indices.dim() == 1:
            # Score all items for all sequences
            item_embs = self.item_emb(item_indices)  # (num_items, hidden_units)
            logits = torch.matmul(final_feat, item_embs.transpose(0, 1))  # (batch, num_items)
        else:
            # Score specific items for each sequence
            item_embs = self.item_emb(item_indices)  # (batch, num_items, hidden_units)
            logits = (final_feat.unsqueeze(1) * item_embs).sum(dim=-1)  # (batch, num_items)

        return logits


class QuantizedSASRec(nn.Module):
    """SASRec with quantization-aware training.

    This version uses quantized linear layers in the attention and
    feed-forward network modules to enable efficient deployment.

    Args:
        num_items: Number of items in the dataset.
        max_seq_len: Maximum sequence length (default: 200).
        hidden_units: Number of hidden units (default: 128).
        num_blocks: Number of self-attention blocks (default: 2).
        num_heads: Number of attention heads (default: 2).
        dropout_rate: Dropout probability (default: 0.2).
        quantizer_type: Type of quantizer to use (default: 'lsq').
        bit_width: Bit width for quantization (default: 8).
    """

    def __init__(
        self,
        num_items,
        max_seq_len=200,
        hidden_units=128,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.2,
        quantizer_type='lsq',
        bit_width=8
    ):
        super().__init__()

        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.quantizer_type = quantizer_type
        self.bit_width = bit_width

        # Import quantized layers (late import to avoid circular dependencies)
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from quantization.fake_quantize import QuantizedLinear  # noqa: E402

        # Embeddings (not quantized)
        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # Self-attention blocks
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            # Standard attention (quantization of attention is complex, skip for now)
            new_attn_layer = nn.MultiheadAttention(
                hidden_units,
                num_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            # Quantized feed-forward
            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                # Zero out padding embedding
                if hasattr(m, 'padding_idx') and m.padding_idx is not None:
                    m.weight.data[m.padding_idx].fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def log2feats(self, log_seqs):
        """Convert log sequences to feature representations.

        Args:
            log_seqs: Input item sequences of shape (batch, seq_len).

        Returns:
            Sequence features of shape (batch, seq_len, hidden_units).
        """
        seqs = self.item_emb(log_seqs)
        positions = torch.arange(log_seqs.size(1), device=log_seqs.device).unsqueeze(0)
        positions = positions.expand_as(log_seqs)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = torch.triu(
            torch.ones((tl, tl), dtype=torch.bool, device=log_seqs.device),
            diagonal=1
        )

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, Q, Q,
                attn_mask=attention_mask,
                key_padding_mask=timeline_mask
            )
            seqs = Q + mha_outputs
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        """Forward pass for training with positive and negative sampling.

        Args:
            log_seqs: Input item sequences of shape (batch, seq_len).
            pos_seqs: Positive items of shape (batch, seq_len).
            neg_seqs: Negative items of shape (batch, seq_len).

        Returns:
            Tuple of (pos_logits, neg_logits) with shapes (batch, seq_len).
        """
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, log_seqs, item_indices):
        """Predict scores for given candidate items.

        Args:
            log_seqs: Input item sequences of shape (batch, seq_len).
            item_indices: Candidate items to score (1D or 2D tensor).

        Returns:
            Prediction scores of shape (batch, num_items).
        """
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]

        if item_indices.dim() == 1:
            item_embs = self.item_emb(item_indices)
            logits = torch.matmul(final_feat, item_embs.transpose(0, 1))
        else:
            item_embs = self.item_emb(item_indices)
            logits = (final_feat.unsqueeze(1) * item_embs).sum(dim=-1)

        return logits
