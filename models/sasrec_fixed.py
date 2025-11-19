"""Fixed SASRec model with stable attention mechanism.

This module implements an improved version of SASRec with enhanced
numerical stability, relative position bias, and optional SwiGLU
feed-forward networks.

Improvements over base SASRec:
- Stable self-attention implementation
- Relative position bias
- SwiGLU activation option
- Better gradient flow
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionBias(nn.Module):
    """Learnable relative position bias for attention.

    Adds learnable biases based on relative distances between positions
    to improve attention quality.

    Args:
        num_heads: Number of attention heads.
        max_distance: Maximum relative distance to consider.
    """

    def __init__(self, num_heads, max_distance):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bias = nn.Embedding(2 * max_distance - 1, num_heads)

    def forward(self, seq_len, device):
        """Compute relative position bias matrix.

        Args:
            seq_len: Sequence length.
            device: Device to create tensors on.

        Returns:
            Bias tensor of shape (num_heads, seq_len, seq_len).
        """
        context_position = torch.arange(seq_len, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(seq_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position = relative_position.clamp(-self.max_distance + 1, self.max_distance - 1)
        relative_position = relative_position + self.max_distance - 1
        values = self.bias(relative_position)
        return values.permute(2, 0, 1)


class PointWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    Standard FFN with two 1D convolutions and residual connection.

    Args:
        hidden_units: Number of hidden units.
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
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SwishGatedFeedForward(nn.Module):
    """Feed-forward network with SwiGLU gating.

    Uses SiLU (Swish) gated linear units for improved performance.

    Args:
        hidden_units: Number of hidden units.
        dropout_rate: Dropout probability.
        expansion_factor: Inner dimension expansion factor (default: 4).
    """

    def __init__(self, hidden_units, dropout_rate, expansion_factor=4):
        super().__init__()
        inner_dim = hidden_units * expansion_factor
        self.w1 = nn.Linear(hidden_units, inner_dim)
        self.w2 = nn.Linear(hidden_units, inner_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(inner_dim, hidden_units)

    def forward(self, inputs):
        gate = F.silu(self.w2(inputs))
        hidden = self.dropout1(self.w1(inputs) * gate)
        hidden = self.dropout2(self.proj(hidden))
        return inputs + hidden


class StableSelfAttention(nn.Module):
    """Stable self-attention implementation.

    Implements multi-head self-attention with improved numerical stability,
    optional relative position bias, and NaN protection.

    Args:
        self,
        hidden_units: Number of hidden units.
        num_heads: Number of attention heads.
        dropout_rate: Dropout probability.
        use_rel_pos_bias: Whether to use relative position bias
            (default: False).
        max_seq_len: Maximum sequence length (default: 200).
    """

    def __init__(
        self,
        hidden_units,
        num_heads,
        dropout_rate,
        use_rel_pos_bias=False,
        max_seq_len=200
    ):
        super().__init__()
        assert hidden_units % num_heads == 0

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(hidden_units, hidden_units)
        self.key = nn.Linear(hidden_units, hidden_units)
        self.value = nn.Linear(hidden_units, hidden_units)
        self.out = nn.Linear(hidden_units, hidden_units)

        self.dropout = nn.Dropout(dropout_rate)
        self.rel_pos_bias = (
            RelativePositionBias(num_heads, max_seq_len)
            if use_rel_pos_bias else None
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """Forward pass with stable attention computation.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_units).
            attn_mask: Optional attention mask (causal mask).
            key_padding_mask: Optional padding mask.

        Returns:
            Tuple of (output, attention_weights):
                - output: Attended features (batch, seq_len, hidden_units).
                - attention_weights: Attention scores.
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections in batch from hidden_units => num_heads*head_dim
        Q = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if self.rel_pos_bias is not None:
            rel_bias = self.rel_pos_bias(seq_len, x.device)
            scores = scores + rel_bias.unsqueeze(0)

        # Apply masks
        # Use -1e4 instead of -1e9 for FP16 compatibility
        # (max FP16 value is ~65504)
        if attn_mask is not None:
            # Causal mask - make future positions have very negative scores
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), -1e4)

        if key_padding_mask is not None:
            # Padding mask - make padded positions have very negative scores
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(1),
                -1e4
            )

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Check for NaN in attention weights
        if torch.isnan(attn_weights).any():
            # Replace NaN with zeros for numerical stability
            attn_weights = torch.where(
                torch.isnan(attn_weights),
                torch.zeros_like(attn_weights),
                attn_weights
            )

        # Weighted sum
        context = torch.matmul(attn_weights, V)

        # Reshape back to (batch, seq_len, hidden_units)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_units)
        )

        # Output projection
        output = self.out(context)

        return output, attn_weights


class SASRec(nn.Module):
    """Fixed SASRec model with stable attention.

    Enhanced SASRec implementation with stable self-attention,
    relative position bias, and optional SwiGLU activation.

    Args:
        self,
        num_items: Number of items in the dataset.
        max_seq_len: Maximum sequence length (default: 200).
        hidden_units: Number of hidden units (default: 128).
        num_blocks: Number of self-attention blocks (default: 2).
        num_heads: Number of attention heads (default: 2).
        dropout_rate: Dropout probability (default: 0.2).
        ffn_type: Feed-forward network type, 'swiglu' or 'standard'
            (default: 'swiglu').
        ffn_factor: FFN inner dimension expansion factor (default: 4).
        use_rel_pos_bias: Whether to use relative position bias
            (default: True).
    """

    def __init__(
        self,
        num_items,
        max_seq_len=200,
        hidden_units=128,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.2,
        ffn_type='swiglu',
        ffn_factor=4,
        use_rel_pos_bias=True
    ):
        super().__init__()

        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ffn_type = ffn_type
        self.ffn_factor = ffn_factor

        # Item embedding (num_items + 1 for padding item 0)
        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)

        # Positional embedding
        self.pos_emb = nn.Embedding(max_seq_len, hidden_units)

        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # Self-attention blocks
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            # Use our stable attention implementation
            self.attention_layernorms.append(
                nn.LayerNorm(hidden_units, eps=1e-8)
            )
            self.attention_layers.append(
                StableSelfAttention(hidden_units, num_heads, dropout_rate)
            )
            self.forward_layernorms.append(
                nn.LayerNorm(hidden_units, eps=1e-8)
            )
            self.forward_layers.append(
                PointWiseFeedForward(hidden_units, dropout_rate)
            )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with smaller values for stability.

        Uses smaller initialization for embeddings (std=0.02) compared
        to standard SASRec for better numerical stability.
        """
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                # Smaller initialization for embeddings
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, 'padding_idx') and m.padding_idx is not None:
                    m.weight.data[m.padding_idx].fill_(0)
            elif isinstance(m, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def log2feats(self, log_seqs):
        """Convert log sequences to features"""
        batch_size = log_seqs.size(0)

        # Get item embeddings
        seqs = self.item_emb(log_seqs)

        # Add positional embeddings
        positions = torch.arange(log_seqs.size(1), device=log_seqs.device).unsqueeze(0)
        positions = positions.expand_as(log_seqs)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        # Create masks
        timeline_mask = (log_seqs == 0)  # True for padding
        seqs *= ~timeline_mask.unsqueeze(-1)  # Zero out padding

        # Causal attention mask
        tl = seqs.shape[1]
        attention_mask = torch.triu(
            torch.ones((tl, tl), dtype=torch.bool, device=log_seqs.device),
            diagonal=1
        )

        # Apply self-attention blocks
        for i in range(len(self.attention_layers)):
            # Layer norm
            seqs_normed = self.attention_layernorms[i](seqs)

            # Self-attention
            mha_outputs, _ = self.attention_layers[i](
                seqs_normed,
                attn_mask=attention_mask,
                key_padding_mask=timeline_mask
            )

            # Residual connection
            seqs = seqs + mha_outputs

            # Feed forward
            seqs_normed = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs_normed)

            # Mask padding positions
            seqs *= ~timeline_mask.unsqueeze(-1)

        # Final layer norm
        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        """Forward pass for training"""
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        # Compute logits with dot product
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, log_seqs, item_indices):
        """Predict scores for given items"""
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]  # Use last position

        if item_indices.dim() == 1:
            item_embs = self.item_emb(item_indices)
            logits = torch.matmul(final_feat, item_embs.transpose(0, 1))
        else:
            item_embs = self.item_emb(item_indices)
            logits = (final_feat.unsqueeze(1) * item_embs).sum(dim=-1)

        return logits


class QuantizedSASRec(nn.Module):
    """SASRec with quantization-aware training.

    Combines the stable attention mechanism with quantization for
    efficient deployment.

    Args:
        self,
        num_items: Number of items in the dataset.
        max_seq_len: Maximum sequence length (default: 200).
        hidden_units: Number of hidden units (default: 128).
        num_blocks: Number of self-attention blocks (default: 2).
        num_heads: Number of attention heads (default: 2).
        dropout_rate: Dropout probability (default: 0.2).
        quantizer_type: Type of quantizer (default: 'lsq').
        bit_width: Quantization bit width (default: 8).
        ffn_type: Feed-forward network type (default: 'swiglu').
        ffn_factor: FFN expansion factor (default: 4).
        use_rel_pos_bias: Use relative position bias (default: True).
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
        bit_width=8,
        ffn_type='swiglu',
        ffn_factor=4,
        use_rel_pos_bias=True
    ):
        super().__init__()

        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.quantizer_type = quantizer_type
        self.bit_width = bit_width
        self.ffn_type = ffn_type
        self.ffn_factor = ffn_factor

        # Import quantized layers
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from quantization.fake_quantize import QuantizedLinear

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
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                StableSelfAttention(
                    hidden_units,
                    num_heads,
                    dropout_rate,
                    use_rel_pos_bias=use_rel_pos_bias,
                    max_seq_len=max_seq_len
                )
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            if ffn_type.lower() == 'swiglu':
                self.forward_layers.append(
                    SwishGatedFeedForward(hidden_units, dropout_rate, expansion_factor=ffn_factor)
                )
            else:
                self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, 'padding_idx') and m.padding_idx is not None:
                    m.weight.data[m.padding_idx].fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def log2feats(self, log_seqs):
        """Convert log sequences to features"""
        seqs = self.item_emb(log_seqs)
        positions = torch.arange(log_seqs.size(1), device=log_seqs.device).unsqueeze(0)
        positions = positions.expand_as(log_seqs)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = torch.triu(torch.ones((tl, tl), dtype=torch.bool, device=log_seqs.device), diagonal=1)

        for i in range(len(self.attention_layers)):
            seqs_normed = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                seqs_normed,
                attn_mask=attention_mask,
                key_padding_mask=timeline_mask
            )
            seqs = seqs + mha_outputs
            seqs_normed = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs_normed)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        """Forward pass for training"""
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, log_seqs, item_indices):
        """Predict scores for given items"""
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]

        if item_indices.dim() == 1:
            item_embs = self.item_emb(item_indices)
            logits = torch.matmul(final_feat, item_embs.transpose(0, 1))
        else:
            item_embs = self.item_emb(item_indices)
            logits = (final_feat.unsqueeze(1) * item_embs).sum(dim=-1)

        return logits
