"""
Evaluate INT8 LSTM/ESPCN models on CPU and compare with FP32 baselines.

The script reloads the original QAT checkpoints to recover architecture
settings, rebuilds the quantized models, optionally loads the stored INT8
state dicts, and reports both accuracy metrics and forward latency.
"""
import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
torch.backends.quantized.engine = 'fbgemm'

try:
    from torch.quantization import quantize_dynamic, prepare, convert
except ImportError:
    from torch.ao.quantization import quantize_dynamic, prepare, convert

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lstm_classifier import QuantizedLSTMClassifier  # noqa: E402
from models.espcn import QuantizedESPCN  # noqa: E402
from utils.data_loader import (  # noqa: E402
    load_imdb_dataset,
    load_bsd_dataset
)
from utils.metrics import (  # noqa: E402
    calculate_psnr_batch,
    calculate_roc_auc
)


def create_synthetic_text_dataset(num_samples=1000, vocab_size=5000, max_len=128):
    train_data = []
    for _ in range(num_samples):
        seq_len = random.randint(10, max_len)
        indices = torch.randint(1, vocab_size, (seq_len,))
        label = 1 if indices.sum() % 2 == 0 else 0
        train_data.append((indices, seq_len, label))

    test_data = []
    for _ in range(max(1, num_samples // 5)):
        seq_len = random.randint(10, max_len)
        indices = torch.randint(1, vocab_size, (seq_len,))
        label = 1 if indices.sum() % 2 == 0 else 0
        test_data.append((indices, seq_len, label))

    return train_data, test_data, vocab_size


def create_synthetic_sr_dataset(num_samples=100, upscale_factor=3, patch_size=64):
    effective_patch = max(upscale_factor, (patch_size // upscale_factor) * upscale_factor)

    class SyntheticSRDataset(Dataset):
        def __init__(self, count, factor, size):
            self.count = count
            self.factor = factor
            self.size = size

        def __len__(self):
            return self.count

        def __getitem__(self, idx):
            hr_image = torch.rand(3, self.size, self.size)
            lr_size = self.size // self.factor
            lr_image = torch.nn.functional.interpolate(
                hr_image.unsqueeze(0),
                size=(lr_size, lr_size),
                mode='bicubic',
                align_corners=False
            ).squeeze(0)
            return lr_image, hr_image

    return SyntheticSRDataset(num_samples, upscale_factor, effective_patch)


def measure_inference_time(model, dataloader, device, max_batches=50, warmup=5, mode='lstm'):
    model.eval()
    times = []
    cached_batches = []
    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
        cached_batches.append(batch)

    warmup_batches = min(warmup, max(0, len(cached_batches) - 1))

    with torch.no_grad():
        for i, batch in enumerate(cached_batches):
            if mode == 'lstm':
                texts, lengths, _ = batch
                inputs = (texts.to(device), lengths)
            elif mode == 'espcn':
                lr_images, _ = batch
                inputs = (lr_images.to(device),)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            if i < warmup_batches:
                model(*inputs)
                continue

            start = time.time()
            model(*inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)

    if not times:
        return 0.0, 0.0
    return float(np.mean(times)), float(np.std(times))


def evaluate_lstm(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            texts, lengths, batch_labels = batch
            texts = texts.to(device)
            outputs = model(texts, lengths)
            probs = torch.sigmoid(outputs).squeeze(-1)
            preds.append(probs.cpu())
            labels.append(batch_labels)
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    return float(calculate_roc_auc(labels, preds))


def evaluate_espcn(model, dataloader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for lr_images, hr_images in dataloader:
            lr_images = lr_images.to(device)
            outputs = model(lr_images)
            batch_psnr = calculate_psnr_batch(outputs.cpu(), hr_images)
            scores.append(batch_psnr)
    return float(np.mean(scores)) if scores else 0.0


def load_lstm_checkpoint(path: str | Path, device: torch.device):
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


def build_lstm_loader(args):
    """
    Build LSTM evaluation dataloader based on requested dataset.
    """
    if args.dataset == 'imdb':
        _, test_dataset, vocab_size = load_imdb_dataset(
            data_dir=args.imdb_data_dir,
            max_vocab_size=args.imdb_vocab_size,
            max_len=args.imdb_max_len
        )
        if args.num_test_samples and args.num_test_samples < len(test_dataset):
            indices = list(range(args.num_test_samples))
            dataset = Subset(test_dataset, indices)
        else:
            dataset = test_dataset

        def collate_imdb(batch):
            texts, lengths, labels = zip(*batch)
            texts = torch.stack(texts, dim=0)
            lengths_tensor = torch.tensor(lengths, dtype=torch.long)
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            return texts, lengths_tensor, labels_tensor

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            collate_fn=collate_imdb
        )
        loader.vocab_size = vocab_size  # attach attribute for downstream logging
        return loader

    # default synthetic fallback
    _, test_data, _ = create_synthetic_text_dataset(
        num_samples=max(args.num_test_samples, args.batch_size),
        vocab_size=args.imdb_vocab_size,
        max_len=args.imdb_max_len
    )

    class SimpleDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    dataset = SimpleDataset(test_data)

    def collate_batch(batch):
        sequences, lengths, labels = zip(*batch)
        max_len = max(seq.size(0) for seq in sequences)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, -seq.size(0):] = seq
        lengths_tensor = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return padded, lengths_tensor, labels_tensor

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    loader.vocab_size = args.imdb_vocab_size
    return loader


def build_espcn_loader(args):
    """
    Build ESPCN evaluation dataloader using BSD300 or synthetic data.
    """
    if args.dataset == 'bsd300':
        _, test_dataset = load_bsd_dataset(
            data_dir=args.espcn_data_dir,
            upscale_factor=args.upscale_factor,
            patch_size=args.espcn_patch_size
        )
        if args.num_test_samples and args.num_test_samples < len(test_dataset):
            dataset = Subset(test_dataset, list(range(args.num_test_samples)))
        else:
            dataset = test_dataset
    else:
        dataset = create_synthetic_sr_dataset(
            num_samples=max(args.num_test_samples, args.batch_size),
            upscale_factor=args.upscale_factor,
            patch_size=args.espcn_patch_size
        )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )


def run_lstm_eval(args):
    device = torch.device('cpu')
    base_model, config = load_lstm_checkpoint(args.checkpoint, device)
    dataloader = build_lstm_loader(args)

    print("Evaluating FP32/QAT model...")
    roc_fp32 = evaluate_lstm(base_model, dataloader, device)
    lat_fp32, std_fp32 = measure_inference_time(base_model, dataloader, device, args.latency_batches, mode='lstm')
    print(f"  ROC-AUC: {roc_fp32:.4f}")
    print(f"  Latency: {lat_fp32:.2f} ± {std_fp32:.2f} ms")

    print("Building INT8 model...")
    quant_layers = {nn.Linear, nn.LSTM}
    model_int8 = quantize_dynamic(copy.deepcopy(base_model), quant_layers, dtype=torch.qint8)
    if args.int8_state:
        int8_state = torch.load(args.int8_state, map_location='cpu', weights_only=False)
        missing, unexpected = model_int8.load_state_dict(int8_state, strict=False)
        if missing:
            print(f"[WARN] Missing INT8 keys ({len(missing)})")
        if unexpected:
            print(f"[WARN] Unexpected INT8 keys ({len(unexpected)})")
    model_int8.eval()

    print("Evaluating INT8 model...")
    roc_int8 = evaluate_lstm(model_int8, dataloader, device)
    lat_int8, std_int8 = measure_inference_time(model_int8, dataloader, device, args.latency_batches, mode='lstm')
    print(f"  ROC-AUC: {roc_int8:.4f}")
    print(f"  Latency: {lat_int8:.2f} ± {std_int8:.2f} ms")

    return {
        'model_type': 'lstm',
        'config': config,
        'metrics': {
            'fp32': {'roc_auc': roc_fp32, 'latency_ms': lat_fp32, 'latency_std': std_fp32},
            'int8': {'roc_auc': roc_int8, 'latency_ms': lat_int8, 'latency_std': std_int8}
        }
    }


def run_espcn_eval(args):
    device = torch.device('cpu')
    base_model, config = load_espcn_checkpoint(args.checkpoint, device)
    dataloader = build_espcn_loader(args)

    print("Evaluating QAT model (simulated FP32)...")
    psnr_fp32 = evaluate_espcn(base_model, dataloader, device)
    lat_fp32, std_fp32 = measure_inference_time(base_model, dataloader, device, args.latency_batches, mode='espcn')
    print(f"  PSNR: {psnr_fp32:.2f} dB")
    print(f"  Latency: {lat_fp32:.2f} ± {std_fp32:.2f} ms")

    # Note: QuantizedESPCN is a QAT model with custom quantizers (DSQ/LSQ/etc.)
    # It already performs quantization during forward pass and cannot be
    # converted to PyTorch static INT8 format using prepare/convert.
    # For true INT8 CPU inference, use ONNX export with quantization.
    print("\nNote: QAT models already perform quantization during forward pass.")
    print("For true INT8 CPU inference, consider ONNX export with INT8 optimization.")
    print("Reporting QAT model metrics as 'int8' for comparison purposes.\n")

    # Use the QAT model as-is (it already quantizes weights/activations)
    model_int8 = copy.deepcopy(base_model)
    model_int8.eval()

    print("Evaluating QAT model (reported as INT8)...")
    psnr_int8 = evaluate_espcn(model_int8, dataloader, device)
    lat_int8, std_int8 = measure_inference_time(model_int8, dataloader, device, args.latency_batches, mode='espcn')
    print(f"  PSNR: {psnr_int8:.2f} dB")
    print(f"  Latency: {lat_int8:.2f} ± {std_int8:.2f} ms")

    return {
        'model_type': 'espcn',
        'config': config,
        'note': 'QAT model with custom quantizers - not PyTorch static INT8',
        'metrics': {
            'fp32': {'psnr': psnr_fp32, 'latency_ms': lat_fp32, 'latency_std': std_fp32},
            'qat_int8': {'psnr': psnr_int8, 'latency_ms': lat_int8, 'latency_std': std_int8}
        }
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LSTM/ESPCN INT8 models on CPU.")
    parser.add_argument('--model_type', required=True, choices=['lstm', 'espcn'])
    parser.add_argument('--checkpoint', required=True, help='Path to the QAT checkpoint (.pt)')
    parser.add_argument('--int8_state', default=None, help='Optional INT8 state dict to load')
    parser.add_argument('--num_test_samples', type=int, default=1000, help='Number of samples for evaluation (subset if dataset provides more)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation/latency')
    parser.add_argument('--latency_batches', type=int, default=20, help='Batches for latency benchmark')
    parser.add_argument('--calibration_batches', type=int, default=10,
                        help='Number of batches for calibration (ESPCN only)')
    parser.add_argument('--dataset', default='synthetic',
                        help='Dataset selection per model: lstm -> [synthetic|imdb], espcn -> [synthetic|bsd300]')
    parser.add_argument('--imdb_data_dir', default='./data/imdb', help='Path to IMDB dataset root')
    parser.add_argument('--imdb_vocab_size', type=int, default=20000, help='Max vocabulary size for IMDB loader')
    parser.add_argument('--imdb_max_len', type=int, default=256, help='Max token count per IMDB sample')
    parser.add_argument('--espcn_data_dir', default='./data/BSD300', help='Path to BSD dataset for ESPCN')
    parser.add_argument('--espcn_patch_size', type=int, default=96, help='Patch size for BSD super-resolution evaluation')
    parser.add_argument('--upscale_factor', type=int, default=3, help='Super-resolution upscale factor')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--output', default=None, help='Optional JSON file to store results')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_type == 'lstm':
        summary = run_lstm_eval(args)
    else:
        summary = run_espcn_eval(args)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved metrics to {args.output}")


if __name__ == '__main__':
    main()
