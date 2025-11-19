"""
Convert fake quantized models to real INT8 using PyTorch quantization
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
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.lstm_classifier import LSTMClassifier, QuantizedLSTMClassifier
from models.espcn import ESPCN, QuantizedESPCN
from models.sasrec import SASRec, QuantizedSASRec
from utils.datasets import create_synthetic_text_dataset, create_synthetic_sr_dataset, create_synthetic_sasrec_dataset
from utils.metrics import calculate_roc_auc, calculate_psnr_batch, calculate_ndcg_at_k
from torch.utils.data import DataLoader


def measure_inference_time(model, dataloader, device, num_iterations=100):
    """
    Measure inference time

    Args:
        model: Model to benchmark
        dataloader: Data loader
        device: Device
        num_iterations: Number of iterations

    Returns:
        Average inference time in ms
    """
    model.eval()
    times = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_iterations:
                break

            # Move data to device
            if isinstance(batch[0], torch.Tensor):
                if len(batch) == 3:  # LSTM or SASRec
                    data = batch[0].to(device)
                else:  # ESPCN
                    data = batch[0].to(device)

                # Warm up
                if i < 10:
                    _ = model(data)
                    continue

                # Measure time
                start = time.time()
                _ = model(data)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()

                times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


def evaluate_lstm_int8(model, dataloader, device):
    """Evaluate INT8 LSTM model"""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            texts, lengths, labels = batch
            texts = texts.to(device)

            outputs = model(texts, lengths)
            probs = torch.sigmoid(outputs)

            all_predictions.append(probs.cpu())
            all_labels.append(labels)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    roc_auc = calculate_roc_auc(all_labels, all_predictions)
    return roc_auc


def evaluate_espcn_int8(model, dataloader, device):
    """Evaluate INT8 ESPCN model"""
    model.eval()
    psnr_values = []

    with torch.no_grad():
        for batch in dataloader:
            lr_images, hr_images = batch
            lr_images = lr_images.to(device)

            outputs = model(lr_images)

            # Calculate PSNR
            batch_psnr = calculate_psnr_batch(outputs.cpu(), hr_images)
            psnr_values.append(batch_psnr)

    avg_psnr = np.mean(psnr_values)
    return avg_psnr


def convert_lstm_to_int8(model_path, quantization_method, device='cpu'):
    """
    Convert LSTM model to INT8

    Args:
        model_path: Path to trained model
        quantization_method: 'lsq' or 'pact'
        device: Device (should be 'cpu' for INT8)

    Returns:
        Quantized model
    """
    print(f"Loading model from {model_path}...")

    # Load checkpoint
    # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint['args']

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

    # Convert using PyTorch dynamic quantization (for LSTM)
    print("Converting to INT8 using dynamic quantization...")
    model_int8 = quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8
    )

    return model_int8


def convert_espcn_to_int8(model_path, quantization_method, device='cpu'):
    """
    Convert ESPCN model to INT8

    Args:
        model_path: Path to trained model
        quantization_method: 'lsq' or 'pact'
        device: Device

    Returns:
        Quantized model
    """
    print(f"Loading model from {model_path}...")

    # Load checkpoint
    # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint['args']

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

    # Convert using PyTorch static quantization
    print("Converting to INT8 using static quantization...")

    # Set quantization config
    try:
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    except:
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')

    # Prepare for quantization
    model_prepared = prepare(model)

    # Convert to INT8
    model_int8 = convert(model_prepared)

    return model_int8


def convert_sasrec_to_int8(model_path, quantization_method, device='cpu'):
    """
    Convert SASRec model to INT8

    Args:
        model_path: Path to trained model
        quantization_method: 'lsq' or 'pact'
        device: Device

    Returns:
        Quantized model
    """
    print(f"Loading model from {model_path}...")

    # Load checkpoint
    # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint['args']

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

    # Convert using dynamic quantization
    print("Converting to INT8 using dynamic quantization...")
    model_int8 = quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    return model_int8


def main(args):
    device = torch.device('cpu')  # INT8 inference is typically done on CPU
    print(f"Using device: {device}")

    # Load and convert model
    if args.model_type == 'lstm':
        model_int8 = convert_lstm_to_int8(args.model_path, args.quantization, device)

        # Create test dataset
        print("Creating test dataset...")
        _, test_data, vocab_size = create_synthetic_text_dataset(
            num_samples=args.num_test_samples,
            vocab_size=20000,
            max_len=128
        )

        class SimpleDataset:
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        test_dataset = SimpleDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Evaluate
        print("Evaluating INT8 model...")
        roc_auc = evaluate_lstm_int8(model_int8, test_loader, device)
        print(f"INT8 Model ROC-AUC: {roc_auc:.4f}")

        # Measure inference time
        print("Measuring inference time...")
        avg_time, std_time = measure_inference_time(model_int8, test_loader, device)
        print(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")

    elif args.model_type == 'espcn':
        model_int8 = convert_espcn_to_int8(args.model_path, args.quantization, device)

        # Create test dataset
        print("Creating test dataset...")
        test_dataset = create_synthetic_sr_dataset(num_samples=200, upscale_factor=3, patch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Evaluate
        print("Evaluating INT8 model...")
        psnr = evaluate_espcn_int8(model_int8, test_loader, device)
        print(f"INT8 Model PSNR: {psnr:.2f} dB")

        # Measure inference time
        print("Measuring inference time...")
        avg_time, std_time = measure_inference_time(model_int8, test_loader, device)
        print(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")

    elif args.model_type == 'sasrec':
        model_int8 = convert_sasrec_to_int8(args.model_path, args.quantization, device)

        print("INT8 conversion completed for SASRec")
        print("Note: Full evaluation requires dataset")

    # Save INT8 model
    if args.save_path:
        print(f"Saving INT8 model to {args.save_path}...")
        torch.save(model_int8.state_dict(), args.save_path)
        print("Saved!")

    # Get model size
    if args.save_path:
        model_size_mb = os.path.getsize(args.save_path) / (1024 * 1024)
        print(f"Model size: {model_size_mb:.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert model to INT8')

    parser.add_argument('--model_type', type=str, required=True,
                        choices=['lstm', 'espcn', 'sasrec'],
                        help='Type of model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--quantization', type=str, required=True,
                        choices=['lsq', 'pact', 'dsq', 'adaround', 'apot'],
                        help='Quantization method used during training')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save INT8 model')
    parser.add_argument('--num_test_samples', type=int, default=1000,
                        help='Number of test samples for evaluation')

    args = parser.parse_args()

    main(args)
