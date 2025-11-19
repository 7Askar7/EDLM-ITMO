"""
Simple INT8 conversion script without torchtext dependency
"""
import torch
import torch.nn as nn
try:
    from torch.quantization import quantize_dynamic, prepare, convert
except ImportError:
    from torch.ao.quantization import quantize_dynamic, prepare, convert
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.lstm_classifier import QuantizedLSTMClassifier
from models.espcn import QuantizedESPCN
from models.sasrec import QuantizedSASRec


def convert_lstm_to_int8(model_path, quantization_method, save_path, device='cpu'):
    """Convert LSTM model to INT8"""
    print(f"Loading LSTM model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint.get('args', checkpoint.get('config', {}))

    # Create model
    model = QuantizedLSTMClassifier(
        vocab_size=args.get('vocab_size', 20000),
        embedding_dim=args.get('embedding_dim', 128),
        hidden_dim=args.get('hidden_dim', 256),
        num_layers=args.get('num_layers', 2),
        dropout=args.get('dropout', 0.5),
        bidirectional=args.get('bidirectional', True),
        quantizer_type=quantization_method,
        bit_width=8
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get original size
    torch.save(model.state_dict(), save_path + '.temp')
    fp32_size = os.path.getsize(save_path + '.temp') / (1024 * 1024)
    os.remove(save_path + '.temp')

    # Convert to INT8 using dynamic quantization
    print("Converting to INT8 using dynamic quantization...")
    model_int8 = quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8
    )

    # Save INT8 model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model_int8.state_dict(), save_path)
    int8_size = os.path.getsize(save_path) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"LSTM-{quantization_method.upper()} Conversion Complete!")
    print(f"{'='*60}")
    print(f"FP32 Model Size: {fp32_size:.2f} MB")
    print(f"INT8 Model Size: {int8_size:.2f} MB")
    print(f"Size Reduction: {((fp32_size - int8_size) / fp32_size * 100):.1f}%")
    print(f"Saved to: {save_path}")
    print(f"{'='*60}")

    return model_int8


def convert_espcn_to_int8(model_path, quantization_method, save_path, device='cpu'):
    """Convert ESPCN model to INT8"""
    print(f"Loading ESPCN model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint.get('args', checkpoint.get('config', {}))

    # Create model
    model = QuantizedESPCN(
        upscale_factor=args.get('upscale_factor', 3),
        num_channels=3,
        feature_channels=args.get('feature_channels', 64),
        quantizer_type=quantization_method,
        bit_width=8
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get original size
    torch.save(model.state_dict(), save_path + '.temp')
    fp32_size = os.path.getsize(save_path + '.temp') / (1024 * 1024)
    os.remove(save_path + '.temp')

    # Convert to INT8 using static quantization
    print("Converting to INT8 using static quantization...")
    try:
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    except:
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')

    model_prepared = prepare(model)
    model_int8 = convert(model_prepared)

    # Save INT8 model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model_int8.state_dict(), save_path)
    int8_size = os.path.getsize(save_path) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"ESPCN-{quantization_method.upper()} Conversion Complete!")
    print(f"{'='*60}")
    print(f"FP32 Model Size: {fp32_size:.2f} MB")
    print(f"INT8 Model Size: {int8_size:.2f} MB")
    print(f"Size Reduction: {((fp32_size - int8_size) / fp32_size * 100):.1f}%")
    print(f"Saved to: {save_path}")
    print(f"{'='*60}")

    return model_int8


def convert_sasrec_to_int8(model_path, quantization_method, save_path, device='cpu'):
    """Convert SASRec model to INT8"""
    print(f"Loading SASRec model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint.get('args', checkpoint.get('config', {}))

    # Create model
    model = QuantizedSASRec(
        num_items=args.get('num_items', 500),
        max_seq_len=args.get('max_len', 200),
        hidden_units=args.get('hidden_units', 128),
        num_blocks=args.get('num_blocks', 2),
        num_heads=args.get('num_heads', 2),
        dropout_rate=args.get('dropout', 0.2),
        quantizer_type=quantization_method,
        bit_width=8
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get original size
    torch.save(model.state_dict(), save_path + '.temp')
    fp32_size = os.path.getsize(save_path + '.temp') / (1024 * 1024)
    os.remove(save_path + '.temp')

    # Convert to INT8 using dynamic quantization
    print("Converting to INT8 using dynamic quantization...")
    model_int8 = quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    # Save INT8 model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model_int8.state_dict(), save_path)
    int8_size = os.path.getsize(save_path) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"SASRec-{quantization_method.upper()} Conversion Complete!")
    print(f"{'='*60}")
    print(f"FP32 Model Size: {fp32_size:.2f} MB")
    print(f"INT8 Model Size: {int8_size:.2f} MB")
    print(f"Size Reduction: {((fp32_size - int8_size) / fp32_size * 100):.1f}%")
    print(f"Saved to: {save_path}")
    print(f"{'='*60}")

    return model_int8


def main(args):
    device = torch.device('cpu')  # INT8 inference is typically done on CPU
    print(f"Using device: {device}\n")

    if args.model_type == 'lstm':
        convert_lstm_to_int8(args.model_path, args.quantization, args.save_path, device)
    elif args.model_type == 'espcn':
        convert_espcn_to_int8(args.model_path, args.quantization, args.save_path, device)
    elif args.model_type == 'sasrec':
        convert_sasrec_to_int8(args.model_path, args.quantization, args.save_path, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert QAT model to INT8')

    parser.add_argument('--model_type', type=str, required=True,
                        choices=['lstm', 'espcn', 'sasrec'],
                        help='Type of model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--quantization', type=str, required=True,
                        choices=['lsq', 'pact', 'dsq', 'adaround', 'apot'],
                        help='Quantization method used during training')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save INT8 model')

    args = parser.parse_args()
    main(args)
