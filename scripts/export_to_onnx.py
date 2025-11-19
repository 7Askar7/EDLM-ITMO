"""
Export QAT models (LSTM, ESPCN) to ONNX format for deployment.

This script exports trained QAT models to ONNX format, which can then be
optimized for INT8 inference using ONNX Runtime quantization tools.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lstm_classifier import QuantizedLSTMClassifier  # noqa: E402
from models.espcn import QuantizedESPCN  # noqa: E402


def load_lstm_checkpoint(path: str | Path, device: torch.device):
    """Load LSTM QAT model from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get('args', {})
    model = QuantizedLSTMClassifier(
        vocab_size=args.get('vocab_size', 20000),
        embedding_dim=args.get('embedding_dim', 128),
        hidden_dim=args.get('hidden_dim', 256),
        num_layers=args.get('num_layers', 2),
        dropout=args.get('dropout', 0.5),
        bidirectional=args.get('bidirectional', True),
        quantizer_type=args.get('quantization', 'lsq'),
        bit_width=args.get('bit_width', 8)
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, args


def load_espcn_checkpoint(path: str | Path, device: torch.device):
    """Load ESPCN QAT model from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get('args', ckpt.get('config', {}))
    model = QuantizedESPCN(
        upscale_factor=args.get('upscale_factor', 3),
        num_channels=args.get('num_channels', 3),
        feature_channels=args.get('feature_channels', 64),
        quantizer_type=args.get('quantization', 'dsq'),
        bit_width=args.get('bit_width', 8)
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, args


def export_lstm_to_onnx(model, output_path, max_seq_len=128, vocab_size=20000):
    """Export LSTM model to ONNX format."""
    device = next(model.parameters()).device

    batch_size = 1
    dummy_text = torch.randint(1, vocab_size, (batch_size, max_seq_len), dtype=torch.long).to(device)

    class LSTMWrap(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, input_ids):
            return self.base_model(input_ids, None)

    wrapper = LSTMWrap(model)

    torch.onnx.export(
        wrapper,
        dummy_text,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'output': {0: 'batch_size'}
        },
        dynamo=False
    )
    print(f"Exported LSTM to {output_path}")


def export_espcn_to_onnx(model, output_path, upscale_factor=3):
    """Export ESPCN model to ONNX format."""
    device = next(model.parameters()).device

    # Create dummy input (low-resolution image)
    batch_size = 1
    num_channels = 3
    lr_size = 64
    dummy_input = torch.randn(batch_size, num_channels, lr_size, lr_size).to(device)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        },
        dynamo=False
    )
    print(f"Exported ESPCN to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export QAT models to ONNX format.")
    parser.add_argument('--model_type', required=True, choices=['lstm', 'espcn'],
                        help='Type of model to export')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to the QAT checkpoint (.pt)')
    parser.add_argument('--output', required=True,
                        help='Output path for ONNX model (.onnx)')
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help='Maximum sequence length for LSTM (default: 128)')
    parser.add_argument('--vocab_size', type=int, default=20000,
                        help='Vocabulary size for LSTM (default: 20000)')
    parser.add_argument('--upscale_factor', type=int, default=3,
                        help='Upscale factor for ESPCN (default: 3)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cpu')

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.model_type == 'lstm':
        print("Loading LSTM model...")
        model, config = load_lstm_checkpoint(args.checkpoint, device)
        print("Exporting LSTM to ONNX...")
        export_lstm_to_onnx(
            model,
            args.output,
            max_seq_len=args.max_seq_len,
            vocab_size=config.get('vocab_size', args.vocab_size)
        )
    else:
        print("Loading ESPCN model...")
        model, config = load_espcn_checkpoint(args.checkpoint, device)
        print("Exporting ESPCN to ONNX...")
        export_espcn_to_onnx(
            model,
            args.output,
            upscale_factor=config.get('upscale_factor', args.upscale_factor)
        )

    print(f"\nSuccessfully exported {args.model_type} model to {args.output}")
    print("\nNext steps:")
    print("1. Quantize ONNX model to INT8 using ONNX Runtime")
    print("2. Run benchmark with: python scripts/benchmark_onnx.py")


if __name__ == '__main__':
    main()
