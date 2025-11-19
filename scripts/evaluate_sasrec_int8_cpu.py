"""
Evaluate a converted SASRec INT8 model on CPU, reporting quality and speed.

This script:
1. Loads the original LSQ checkpoint to recover the architecture config.
2. Builds the QuantizedSASRec model, applies PyTorch dynamic quantization,
   and optionally loads a previously exported INT8 state dict.
3. Runs MovieLens evaluation (all-items and/or 100-negative) on CPU.
4. Measures forward-pass latency on CPU for a handful of batches.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    from torch.quantization import quantize_dynamic
except ImportError:
    from torch.ao.quantization import quantize_dynamic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sasrec_fixed import QuantizedSASRec  # noqa: E402
from utils.data_loader import load_movielens_dataset  # noqa: E402
from utils.evaluation import (  # noqa: E402
    evaluate_100neg,
    evaluate_all_items
)


def _load_checkpoint(path: str | Path, device: torch.device) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def _resolve_num_items(checkpoint: Dict, config: Dict) -> int:
    for key in ('num_items', 'n_items', 'total_items'):
        if key in checkpoint:
            return checkpoint[key]
        if key in config:
            return config[key]
    raise ValueError(
        "Number of items missing in checkpoint metadata. "
        "Re-run training with num_items saved in checkpoint or "
        "pass --num-items."
    )


def _build_model_from_config(
    config: Dict,
    num_items: int,
    quant_method: str,
    bit_width: int
) -> QuantizedSASRec:
    max_seq_len = config.get('max_len', config.get('max_seq_len', 200))
    hidden_units = config.get('hidden_units', 128)
    num_blocks = config.get('num_blocks', 2)
    num_heads = config.get('num_heads', 2)
    dropout = config.get('dropout', config.get('dropout_rate', 0.2))
    ffn_type = config.get('ffn_type', 'swiglu')
    ffn_factor = config.get('ffn_factor', 4)
    use_rel_pos_bias = not config.get('disable_rel_pos_bias', False)

    model = QuantizedSASRec(
        num_items=num_items,
        max_seq_len=max_seq_len,
        hidden_units=hidden_units,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout,
        quantizer_type=quant_method,
        bit_width=bit_width,
        ffn_type=ffn_type,
        ffn_factor=ffn_factor,
        use_rel_pos_bias=use_rel_pos_bias
    )
    return model


def load_int8_model(
    training_checkpoint: str | Path,
    int8_state_dict: Optional[str | Path],
    device: torch.device,
    keep_fp32_model: bool = False
) -> Tuple[torch.nn.Module, int, Dict, Optional[torch.nn.Module]]:
    """Restore architecture from training checkpoint and attach INT8 weights."""
    training_checkpoint = Path(training_checkpoint)
    ckpt = _load_checkpoint(training_checkpoint, device)
    config = ckpt.get('args', ckpt.get('config', {}))

    quant_method = (config.get('quantization') or 'lsq').lower()
    bit_width = config.get('bit_width', 8)

    num_items = _resolve_num_items(ckpt, config)

    base_model = _build_model_from_config(
        config,
        num_items,
        quant_method,
        bit_width
    )
    state_dict = ckpt.get('model_state_dict') or ckpt.get('state_dict')
    missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading training checkpoint ({len(missing)}).")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading training checkpoint ({len(unexpected)}).")
    base_model.eval()

    model_int8 = quantize_dynamic(
        base_model,
        {nn.Linear},
        dtype=torch.qint8
    )
    model_int8.eval()

    if int8_state_dict:
        int8_state_dict = Path(int8_state_dict)
        print(f"Loading INT8 state dict from {int8_state_dict}...")
        int8_weights = torch.load(int8_state_dict, map_location='cpu', weights_only=False)
        missing, unexpected = model_int8.load_state_dict(int8_weights, strict=False)
        if missing:
            print(f"[WARN] Missing keys in INT8 state dict ({len(missing)}).")
        if unexpected:
            print(f"[WARN] Unexpected keys in INT8 state dict ({len(unexpected)}).")

    fp32_model = base_model.to(device) if keep_fp32_model else None
    return model_int8.to(device), num_items, config, fp32_model


def measure_forward_latency(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 64,
    num_batches: int = 50,
    warmup_batches: int = 5
) -> Tuple[float, float]:
    """
    Measure forward-pass latency (ms) over a subset of the dataset.

    Uses the training-style forward pass (log_seq, pos, neg) since it exercises
    the same blocks as ranking inference.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    timings = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches + warmup_batches:
                break

            log_seqs, pos_seqs, neg_seqs = batch
            log_seqs = log_seqs.to(device)
            pos_seqs = pos_seqs.to(device)
            neg_seqs = neg_seqs.to(device)

            if batch_idx < warmup_batches:
                _ = model(log_seqs, pos_seqs, neg_seqs)
                continue

            start = time.perf_counter()
            _ = model(log_seqs, pos_seqs, neg_seqs)
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # to ms

    if not timings:
        return 0.0, 0.0

    return float(np.mean(timings)), float(np.std(timings))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SASRec INT8 model on CPU (quality & speed)."
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to the original LSQ checkpoint with model args.'
    )
    parser.add_argument(
        '--int8_state',
        default=None,
        help='Optional INT8 state dict produced by convert_sasrec_lsq_int8.py.'
    )
    parser.add_argument(
        '--data_dir',
        default='./data/ml-25m',
        help='MovieLens data directory (use ml-25m to match production runs).'
    )
    parser.add_argument(
        '--min_rating',
        type=float,
        default=4.0,
        help='Minimum rating threshold.'
    )
    parser.add_argument(
        '--mode',
        choices=['all', '100neg', 'both'],
        default='100neg',
        help='Evaluation mode to run.'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Top-k for NDCG/HR.'
    )
    parser.add_argument(
        '--num_negatives',
        type=int,
        default=100,
        help='Negative samples for 100-neg evaluation.'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=2000,
        help='Number of test samples to use (subset). Set 0 to use full dataset.'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Optional path to save metrics as JSON.'
    )
    parser.add_argument(
        '--latency_batches',
        type=int,
        default=50,
        help='Number of batches for latency benchmark.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for latency benchmark.'
    )
    parser.add_argument(
        '--compare_fp32',
        action='store_true',
        help='Also evaluate the original FP32 checkpoint for reference.'
    )
    parser.add_argument(
        '--device',
        choices=['cpu'],
        default='cpu',
        help='Device to run evaluation on (INT8 is CPU-only).'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("Building INT8 model...")
    model_int8, num_items_model, config, fp32_model = load_int8_model(
        training_checkpoint=args.checkpoint,
        int8_state_dict=args.int8_state,
        device=device,
        keep_fp32_model=args.compare_fp32
    )
    print("Model ready on CPU.")

    max_len = config.get('max_len', config.get('max_seq_len', 200))

    print("\nLoading MovieLens dataset...")
    _, test_dataset, num_users, num_items_data = load_movielens_dataset(
        data_dir=args.data_dir,
        min_rating=args.min_rating,
        max_len=max_len
    )
    print(f"Users: {num_users}, Items: {num_items_data}, Test samples: {len(test_dataset)}")
    if num_items_data != num_items_model:
        print(f"[WARN] Dataset items ({num_items_data}) != model items ({num_items_model}).")

    eval_dataset = test_dataset
    subset_size = args.num_samples if args.num_samples else len(test_dataset)
    if subset_size and subset_size < len(test_dataset):
        print(f"Using subset of {subset_size} samples for evaluation to keep runtime manageable.")
        eval_dataset = Subset(test_dataset, list(range(subset_size)))
    actual_samples = len(eval_dataset)

    summary = {
        'checkpoint': args.checkpoint,
        'int8_state': args.int8_state,
        'data_dir': args.data_dir,
        'num_users': num_users,
        'num_items_data': num_items_data,
        'num_items_model': num_items_model,
        'eval_samples': actual_samples
    }

    if args.mode in ('all', 'both'):
        print("\nRunning ALL-ITEMS evaluation on CPU...")
        start = time.perf_counter()
        all_metrics = evaluate_all_items(
            model_int8,
            eval_dataset,
            num_items_model,
            k=args.k,
            device=device,
            num_samples=None,
            verbose=True
        )
        duration = time.perf_counter() - start
        summary['all_items'] = {
            'metrics': all_metrics,
            'seconds': duration,
            'samples_per_sec': all_metrics['num_samples'] / duration if duration > 0 else 0.0
        }
        print(f"All-items NDCG@{args.k}: {all_metrics[f'ndcg@{args.k}']:.4f}")
        print(f"All-items HR@{args.k}: {all_metrics[f'hr@{args.k}']:.4f}")
        print(f"All-items time: {duration/60:.2f} min")

    if args.mode in ('100neg', 'both'):
        print(f"\nRunning {args.num_negatives}-NEG evaluation on CPU...")
        start = time.perf_counter()
        neg_metrics = evaluate_100neg(
            model_int8,
            eval_dataset,
            num_items_model,
            k=args.k,
            num_negatives=args.num_negatives,
            device=device,
            seed=42,
            num_samples=None,
            verbose=True
        )
        duration = time.perf_counter() - start
        summary['neg_sampling'] = {
            'metrics': neg_metrics,
            'seconds': duration,
            'samples_per_sec': neg_metrics['num_samples'] / duration if duration > 0 else 0.0
        }
        print(f"{args.num_negatives}-neg NDCG@{args.k}: {neg_metrics[f'ndcg@{args.k}']:.4f}")
        print(f"{args.num_negatives}-neg HR@{args.k}: {neg_metrics[f'hr@{args.k}']:.4f}")
        print(f"{args.num_negatives}-neg time: {duration/60:.2f} min")

    if args.compare_fp32 and fp32_model is not None:
        print("\nRunning baseline FP32 evaluation on CPU...")
        start = time.perf_counter()
        fp32_metrics = evaluate_100neg(
            fp32_model,
            eval_dataset,
            num_items_model,
            k=args.k,
            num_negatives=args.num_negatives,
            device=device,
            seed=42,
            num_samples=None,
            verbose=True
        )
        duration = time.perf_counter() - start
        summary['fp32_baseline'] = {
            'metrics': fp32_metrics,
            'seconds': duration,
            'samples_per_sec': fp32_metrics['num_samples'] / duration if duration > 0 else 0.0
        }
        print(f"FP32 NDCG@{args.k}: {fp32_metrics[f'ndcg@{args.k}']:.4f}")
        print(f"FP32 HR@{args.k}: {fp32_metrics[f'hr@{args.k}']:.4f}")
        print(f"FP32 eval time: {duration/60:.2f} min")
        fp32_lat_avg, fp32_lat_std = measure_forward_latency(
            fp32_model,
            eval_dataset,
            device=device,
            batch_size=args.batch_size,
            num_batches=args.latency_batches
        )
        summary['fp32_baseline']['latency_ms'] = {
            'avg': fp32_lat_avg,
            'std': fp32_lat_std,
            'batch_size': args.batch_size,
            'batches_measured': args.latency_batches
        }
        print(f"FP32 forward latency: {fp32_lat_avg:.2f} +/- {fp32_lat_std:.2f} ms (batch={args.batch_size})")

    print("\nBenchmarking forward-pass latency...")
    avg_ms, std_ms = measure_forward_latency(
        model_int8,
        eval_dataset,
        device=device,
        batch_size=args.batch_size,
        num_batches=args.latency_batches
    )
    summary['latency_ms'] = {
        'avg': avg_ms,
        'std': std_ms,
        'batch_size': args.batch_size,
        'batches_measured': args.latency_batches
    }
    print(f"Forward latency: {avg_ms:.2f} +/- {std_ms:.2f} ms (batch={args.batch_size})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved metrics to {args.output}")


if __name__ == '__main__':
    main()
