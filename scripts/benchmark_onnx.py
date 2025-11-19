"""
Benchmark and quantize ONNX models to INT8 for CPU inference.

This script:
1. Loads ONNX model exported from QAT checkpoint
2. Applies dynamic or static INT8 quantization using ONNX Runtime
3. Benchmarks both FP32 and INT8 models on CPU
4. Reports accuracy and latency metrics
"""
import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, CalibrationDataReader
    from onnxruntime.quantization.preprocess import quant_pre_process
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("WARNING: ONNX Runtime not available. Please install:")
    print("  pip install onnx onnxruntime")
    raise SystemExit("ONNX Runtime is required to run this script.")


class SyntheticLSTMDataset:
    """Synthetic dataset for LSTM text classification."""

    def __init__(self, num_samples=1000, vocab_size=20000, max_len=128):
        self.data = []
        for _ in range(num_samples):
            seq_len = random.randint(10, max_len)
            indices = np.random.randint(1, vocab_size, size=seq_len, dtype=np.int64)
            label = 1 if indices.sum() % 2 == 0 else 0
            self.data.append((indices, seq_len, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SyntheticSRDataset(Dataset):
    """Synthetic dataset for super-resolution."""

    def __init__(self, num_samples=100, upscale_factor=3, patch_size=64):
        self.count = num_samples
        self.factor = upscale_factor
        # Ensure patch_size is divisible by upscale_factor
        self.lr_size = patch_size // upscale_factor
        self.hr_size = self.lr_size * upscale_factor  # This ensures exact match

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # Create LR image
        lr_image = np.random.rand(3, self.lr_size, self.lr_size).astype(np.float32)
        # Create HR image with exact upscaled size
        hr_image = np.random.rand(3, self.hr_size, self.hr_size).astype(np.float32)
        return lr_image, hr_image


class ESPCNCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader for ESPCN static quantization."""

    def __init__(self, dataset, num_samples=100):
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
        self.current_idx = 0

    def get_next(self):
        if self.current_idx >= self.num_samples:
            return None
        lr_image, _ = self.dataset[self.current_idx]
        self.current_idx += 1
        # Add batch dimension
        return {'input': lr_image[np.newaxis, :, :, :]}


def calculate_psnr_np(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    max_pixel = 1.0
    return float(20 * np.log10(max_pixel / np.sqrt(mse)))


def calculate_roc_auc_np(labels: np.ndarray, preds: np.ndarray) -> float:
    """Calculate ROC-AUC score."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, preds))


def quantize_onnx_dynamic(input_model: str, output_model: str):
    """Apply dynamic quantization to ONNX model."""
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Runtime not available")

    print(f"Applying dynamic INT8 quantization...")
    preprocessed_model = Path(output_model).with_name(
        f"{Path(output_model).stem}_pre.onnx"
    )
    quant_pre_process(
        input_model_path=input_model,
        output_model_path=str(preprocessed_model),
        auto_merge=True,
        optimize_model=True,
        use_fp16=False,
    )

    quantize_dynamic(
        model_input=str(preprocessed_model),
        model_output=output_model,
        weight_type=QuantType.QInt8
    )
    print(f"Saved quantized model to {output_model}")


def quantize_onnx_static(input_model: str, output_model: str, calibration_reader: CalibrationDataReader):
    """Apply static quantization to ONNX model."""
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Runtime not available")

    print(f"Applying static INT8 quantization with calibration...")
    quantize_static(
        model_input=input_model,
        model_output=output_model,
        calibration_data_reader=calibration_reader
    )
    print(f"Saved quantized model to {output_model}")


def benchmark_onnx_lstm(onnx_path: str, num_samples: int = 1000, batch_size: int = 64,
                        max_batches: int = 20) -> Tuple[float, float, float]:
    """Benchmark ONNX LSTM model."""
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Runtime not available")

    # Create synthetic dataset
    dataset = SyntheticLSTMDataset(num_samples=num_samples, vocab_size=20000, max_len=128)

    def collate_batch(batch):
        sequences, lengths, labels = zip(*batch)
        max_len = max(len(seq) for seq in sequences)
        padded = np.zeros((len(batch), max_len), dtype=np.int64)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        lengths_arr = np.array([len(seq) for seq in sequences], dtype=np.int64)
        labels_arr = np.array(labels, dtype=np.float32)
        return padded, lengths_arr, labels_arr

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # Create ONNX session
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Evaluate accuracy
    all_preds = []
    all_labels = []
    for texts, lengths, labels in dataloader:
        inputs = {'input_ids': texts}
        outputs = session.run(None, inputs)[0]
        preds = 1.0 / (1.0 + np.exp(-outputs.squeeze(-1)))  # sigmoid
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    try:
        roc_auc = calculate_roc_auc_np(all_labels, all_preds)
    except ImportError:
        roc_auc = 0.5
        print("Warning: scikit-learn not available, ROC-AUC set to 0.5")

    # Benchmark latency
    times = []
    for i, (texts, lengths, _) in enumerate(dataloader):
        if i >= max_batches:
            break
        inputs = {'input_ids': texts}

        # Warmup
        if i < 2:
            session.run(None, inputs)
            continue

        start = time.time()
        session.run(None, inputs)
        times.append((time.time() - start) * 1000)

    avg_latency = float(np.mean(times)) if times else 0.0
    std_latency = float(np.std(times)) if times else 0.0

    return roc_auc, avg_latency, std_latency


def benchmark_onnx_espcn(onnx_path: str, num_samples: int = 200, batch_size: int = 16,
                         max_batches: int = 20) -> Tuple[float, float, float]:
    """Benchmark ONNX ESPCN model."""
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Runtime not available")

    # Create synthetic dataset
    dataset = SyntheticSRDataset(num_samples=num_samples, upscale_factor=3, patch_size=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Create ONNX session
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Evaluate PSNR
    psnr_scores = []
    for lr_images, hr_images in dataloader:
        inputs = {'input': lr_images.numpy()}
        outputs = session.run(None, inputs)[0]

        # Calculate PSNR for each image in batch
        for i in range(outputs.shape[0]):
            psnr = calculate_psnr_np(outputs[i], hr_images[i].numpy())
            psnr_scores.append(psnr)

    avg_psnr = float(np.mean(psnr_scores)) if psnr_scores else 0.0

    # Benchmark latency
    times = []
    for i, (lr_images, _) in enumerate(dataloader):
        if i >= max_batches:
            break
        inputs = {'input': lr_images.numpy()}

        # Warmup
        if i < 2:
            session.run(None, inputs)
            continue

        start = time.time()
        session.run(None, inputs)
        times.append((time.time() - start) * 1000)

    avg_latency = float(np.mean(times)) if times else 0.0
    std_latency = float(np.std(times)) if times else 0.0

    return avg_psnr, avg_latency, std_latency


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark and quantize ONNX models.")
    parser.add_argument('--model_type', required=True, choices=['lstm', 'espcn'],
                        help='Type of model')
    parser.add_argument('--onnx_model', required=True,
                        help='Path to ONNX model (FP32)')
    parser.add_argument('--quantize', choices=['dynamic', 'static', 'both'], default='dynamic',
                        help='Quantization method')
    parser.add_argument('--output_dir', default='results/onnx',
                        help='Output directory for quantized models')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--latency_batches', type=int, default=20,
                        help='Number of batches for latency benchmark')
    parser.add_argument('--calibration_samples', type=int, default=100,
                        help='Number of samples for calibration (static quantization)')
    parser.add_argument('--output_json', default=None,
                        help='Output JSON file for results')
    return parser.parse_args()


def main():
    args = parse_args()

    if not ONNX_AVAILABLE:
        print("ERROR: ONNX Runtime not installed. Please run:")
        print("  pip install onnx onnxruntime")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'model_type': args.model_type,
        'onnx_model': args.onnx_model,
        'metrics': {}
    }

    # Benchmark FP32 model
    print("\n" + "="*60)
    print("Benchmarking FP32 ONNX model...")
    print("="*60)

    if args.model_type == 'lstm':
        roc_auc, latency, std = benchmark_onnx_lstm(
            args.onnx_model, args.num_samples, args.batch_size, args.latency_batches
        )
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Latency: {latency:.2f} ± {std:.2f} ms")
        results['metrics']['fp32'] = {
            'roc_auc': roc_auc,
            'latency_ms': latency,
            'latency_std': std
        }
    elif args.model_type == 'espcn':
        psnr, latency, std = benchmark_onnx_espcn(
            args.onnx_model, args.num_samples, args.batch_size, args.latency_batches
        )
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  Latency: {latency:.2f} ± {std:.2f} ms")
        results['metrics']['fp32'] = {
            'psnr': psnr,
            'latency_ms': latency,
            'latency_std': std
        }

    # Quantize and benchmark INT8 models
    model_name = Path(args.onnx_model).stem

    if args.quantize in ['dynamic', 'both']:
        print("\n" + "="*60)
        print("Dynamic INT8 Quantization...")
        print("="*60)

        int8_dynamic_path = output_dir / f"{model_name}_int8_dynamic.onnx"
        try:
            quantize_onnx_dynamic(args.onnx_model, str(int8_dynamic_path))

            print("\nBenchmarking INT8 (Dynamic) model...")
            if args.model_type == 'lstm':
                roc_auc, latency, std = benchmark_onnx_lstm(
                    str(int8_dynamic_path), args.num_samples, args.batch_size, args.latency_batches
                )
                print(f"  ROC-AUC: {roc_auc:.4f}")
                print(f"  Latency: {latency:.2f} ± {std:.2f} ms")
                results['metrics']['int8_dynamic'] = {
                    'roc_auc': roc_auc,
                    'latency_ms': latency,
                    'latency_std': std,
                    'model_path': str(int8_dynamic_path)
                }
            elif args.model_type == 'espcn':
                psnr, latency, std = benchmark_onnx_espcn(
                    str(int8_dynamic_path), args.num_samples, args.batch_size, args.latency_batches
                )
                print(f"  PSNR: {psnr:.2f} dB")
                print(f"  Latency: {latency:.2f} ± {std:.2f} ms")
                results['metrics']['int8_dynamic'] = {
                    'psnr': psnr,
                    'latency_ms': latency,
                    'latency_std': std,
                    'model_path': str(int8_dynamic_path)
                }
        except Exception as e:
            print(f"Warning: Dynamic quantization or benchmarking failed: {e}")
            print("Skipping dynamic INT8 results.")
            err_msg = str(e) if str(e) else repr(e)
            results['metrics']['int8_dynamic'] = {'error': err_msg}

    if args.quantize in ['static', 'both'] and args.model_type == 'espcn':
        print("\n" + "="*60)
        print("Static INT8 Quantization...")
        print("="*60)

        int8_static_path = output_dir / f"{model_name}_int8_static.onnx"

        # Create calibration dataset
        calib_dataset = SyntheticSRDataset(
            num_samples=args.calibration_samples,
            upscale_factor=3,
            patch_size=64
        )
        calib_reader = ESPCNCalibrationDataReader(calib_dataset, args.calibration_samples)

        try:
            quantize_onnx_static(args.onnx_model, str(int8_static_path), calib_reader)

            print("\nBenchmarking INT8 (Static) model...")
            psnr, latency, std = benchmark_onnx_espcn(
                str(int8_static_path), args.num_samples, args.batch_size, args.latency_batches
            )
            print(f"  PSNR: {psnr:.2f} dB")
            print(f"  Latency: {latency:.2f} ± {std:.2f} ms")
            results['metrics']['int8_static'] = {
                'psnr': psnr,
                'latency_ms': latency,
                'latency_std': std,
                'model_path': str(int8_static_path)
            }
        except Exception as e:
            print(f"Warning: Static quantization failed: {e}")
            print("Continuing with dynamic quantization only...")

    # Save results
    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for quant_type, metrics in results['metrics'].items():
        print(f"\n{quant_type.upper()}:")
        for key, value in metrics.items():
            if key != 'model_path':
                print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
