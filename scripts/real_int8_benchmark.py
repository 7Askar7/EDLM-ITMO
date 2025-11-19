"""
Real INT8 conversion and CPU benchmark for quantized models
Measures actual speedup and quality after conversion to real INT8
"""
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, prepare_qat, convert
import torch.backends.quantized
import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.lstm_classifier import LSTMClassifier, QuantizedLSTMClassifier
from models.espcn import ESPCN, QuantizedESPCN
from models.sasrec import SASRec, QuantizedSASRec
from utils.data_loader import load_imdb_dataset, load_bsd_dataset, load_movielens_dataset, get_dataloader
from utils.metrics import calculate_roc_auc, calculate_psnr_batch, calculate_ndcg_at_k


class INT8Converter:
    """
    Converts fake quantized models to real INT8 using PyTorch quantization
    """

    def __init__(self, model_type, quantization_method):
        self.model_type = model_type
        self.quantization_method = quantization_method
        self.device = torch.device('cpu')  # INT8 optimized for CPU

    def load_model(self, checkpoint_path):
        """Load trained model from checkpoint"""
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")

        # Load appropriate model architecture
        if self.model_type == 'lstm':
            if self.quantization_method == 'none':
                model = LSTMClassifier(
                    vocab_size=20000,
                    embedding_dim=128,
                    hidden_dim=256,
                    output_dim=1,
                    num_layers=2,
                    bidirectional=True,
                    dropout=0.5
                )
            else:
                model = QuantizedLSTMClassifier(
                    vocab_size=20000,
                    embedding_dim=128,
                    hidden_dim=256,
                    output_dim=1,
                    num_layers=2,
                    bidirectional=True,
                    dropout=0.5,
                    quantization_method=self.quantization_method
                )

        elif self.model_type == 'espcn':
            if self.quantization_method == 'none':
                model = ESPCN(upscale_factor=3)
            else:
                model = QuantizedESPCN(
                    upscale_factor=3,
                    quantization_method=self.quantization_method
                )

        elif self.model_type == 'sasrec':
            if self.quantization_method == 'none':
                model = SASRec(
                    item_num=3500,
                    maxlen=200,
                    hidden_units=128,
                    num_blocks=2,
                    num_heads=2,
                    dropout_rate=0.2
                )
            else:
                model = QuantizedSASRec(
                    item_num=3500,
                    maxlen=200,
                    hidden_units=128,
                    num_blocks=2,
                    num_heads=2,
                    dropout_rate=0.2,
                    quantization_method=self.quantization_method
                )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Load weights
        # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    def convert_to_int8(self, model, calibration_data=None):
        """
        Convert fake quantized model to real INT8

        Args:
            model: Trained model with fake quantization
            calibration_data: Data for calibration (optional)

        Returns:
            Quantized model
        """
        print(f"[INFO] Converting to real INT8...")

        if self.model_type == 'lstm':
            # Dynamic quantization for LSTM (RNN models work better with dynamic)
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                qconfig_spec={
                    nn.Linear: torch.quantization.default_dynamic_qconfig,
                    nn.LSTM: torch.quantization.default_dynamic_qconfig,
                },
                dtype=torch.qint8
            )

        elif self.model_type in ['espcn', 'sasrec']:
            # Static quantization for CNN and Transformer
            backend = "fbgemm" if torch.backends.quantized.engine == 'fbgemm' else "qnnpack"
            model.qconfig = torch.quantization.get_default_qconfig(backend)

            # Prepare model for quantization
            model = torch.quantization.prepare(model, inplace=False)

            # Calibration step
            if calibration_data is not None:
                print("[INFO] Running calibration...")
                with torch.no_grad():
                    for batch in tqdm(calibration_data, desc="Calibration"):
                        if self.model_type == 'espcn':
                            lr_images, _ = batch
                            _ = model(lr_images.to(self.device))
                        elif self.model_type == 'sasrec':
                            seq, labels = batch
                            _ = model(seq.to(self.device))

            # Convert to quantized model
            quantized_model = torch.quantization.convert(model, inplace=False)

        else:
            raise NotImplementedError(f"Quantization not implemented for {self.model_type}")

        return quantized_model

    def benchmark_inference(self, model, dataloader, num_iterations=100):
        """
        Benchmark inference speed on CPU

        Args:
            model: Model to benchmark
            dataloader: Test dataloader
            num_iterations: Number of iterations to average

        Returns:
            dict: Timing results
        """
        print(f"[INFO] Benchmarking inference speed (CPU)...")
        model.eval()
        model = model.to(self.device)

        # Warmup
        print("[INFO] Warming up...")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 10:
                    break
                if self.model_type == 'lstm':
                    texts, lengths, _ = batch
                    _ = model(texts.to(self.device))
                elif self.model_type == 'espcn':
                    lr_images, _ = batch
                    _ = model(lr_images.to(self.device))
                elif self.model_type == 'sasrec':
                    seq, _ = batch
                    _ = model(seq.to(self.device))

        # Actual benchmark
        times = []
        print(f"[INFO] Running {num_iterations} iterations...")

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=num_iterations)):
                if i >= num_iterations:
                    break

                if self.model_type == 'lstm':
                    texts, lengths, _ = batch
                    data = texts.to(self.device)
                elif self.model_type == 'espcn':
                    lr_images, _ = batch
                    data = lr_images.to(self.device)
                elif self.model_type == 'sasrec':
                    seq, _ = batch
                    data = seq.to(self.device)

                start_time = time.perf_counter()
                _ = model(data)
                end_time = time.perf_counter()

                times.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        times = np.array(times)
        results = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'median_ms': np.median(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'min_ms': np.min(times),
            'max_ms': np.max(times)
        }

        return results

    def evaluate_quality(self, model, dataloader):
        """
        Evaluate model quality after INT8 conversion

        Args:
            model: Quantized model
            dataloader: Test dataloader

        Returns:
            dict: Quality metrics
        """
        print(f"[INFO] Evaluating model quality...")
        model.eval()
        model = model.to(self.device)

        if self.model_type == 'lstm':
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Evaluating"):
                    texts, lengths, labels = batch
                    texts = texts.to(self.device)

                    outputs = model(texts)
                    preds = torch.sigmoid(outputs).cpu().numpy()

                    all_preds.extend(preds.flatten())
                    all_labels.extend(labels.numpy())

            auc = calculate_roc_auc(all_labels, all_preds)
            return {'roc_auc': auc}

        elif self.model_type == 'espcn':
            psnr_values = []

            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Evaluating"):
                    lr_images, hr_images = batch
                    lr_images = lr_images.to(self.device)
                    hr_images = hr_images.to(self.device)

                    sr_images = model(lr_images)
                    psnr = calculate_psnr_batch(sr_images, hr_images)
                    psnr_values.append(psnr)

            return {'psnr': np.mean(psnr_values)}

        elif self.model_type == 'sasrec':
            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Evaluating"):
                    seq, labels = batch
                    seq = seq.to(self.device)

                    outputs = model(seq)
                    predictions = outputs[:, -1, :]

                    all_predictions.append(predictions.cpu())
                    all_labels.append(labels)

            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            ndcg = calculate_ndcg_at_k(all_predictions, all_labels, k=10)
            return {'ndcg@10': ndcg}

    def get_model_size(self, model):
        """
        Calculate model size in MB

        Args:
            model: PyTorch model

        Returns:
            float: Model size in MB
        """
        # Save model temporarily
        temp_path = 'temp_model.pth'
        torch.save(model.state_dict(), temp_path)

        # Get file size
        size_bytes = os.path.getsize(temp_path)
        size_mb = size_bytes / (1024 * 1024)

        # Clean up
        os.remove(temp_path)

        return size_mb


def run_full_benchmark(args):
    """
    Run complete benchmark for all models and methods
    """
    results = []

    # Load datasets
    print("[INFO] Loading datasets...")
    if args.model_type == 'lstm':
        train_data, test_data = load_imdb_dataset()
        test_loader = get_dataloader(test_data, batch_size=args.batch_size, shuffle=False)
        calibration_loader = get_dataloader(train_data, batch_size=args.batch_size, shuffle=True)
    elif args.model_type == 'espcn':
        train_data, test_data = load_bsd_dataset()
        test_loader = get_dataloader(test_data, batch_size=args.batch_size, shuffle=False)
        calibration_loader = get_dataloader(train_data, batch_size=args.batch_size, shuffle=True)
    elif args.model_type == 'sasrec':
        train_data, test_data = load_movielens_dataset()
        test_loader = get_dataloader(test_data, batch_size=args.batch_size, shuffle=False)
        calibration_loader = get_dataloader(train_data, batch_size=args.batch_size, shuffle=True)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Process each quantization method
    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Processing: {args.model_type.upper()} - {method.upper()}")
        print(f"{'='*60}")

        # Initialize converter
        converter = INT8Converter(args.model_type, method)

        # Find checkpoint
        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f"{args.model_type}_{method}_best.pt"
        )

        if not os.path.exists(checkpoint_path):
            print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
            continue

        try:
            # Load FP32 model
            fp32_model = converter.load_model(checkpoint_path)

            # Benchmark FP32
            print("\n[1/4] Benchmarking FP32 model...")
            fp32_timing = converter.benchmark_inference(fp32_model, test_loader, args.num_iterations)
            fp32_quality = converter.evaluate_quality(fp32_model, test_loader)
            fp32_size = converter.get_model_size(fp32_model)

            # Convert to INT8
            print("\n[2/4] Converting to INT8...")
            int8_model = converter.convert_to_int8(fp32_model, calibration_loader)

            # Benchmark INT8
            print("\n[3/4] Benchmarking INT8 model...")
            int8_timing = converter.benchmark_inference(int8_model, test_loader, args.num_iterations)
            int8_quality = converter.evaluate_quality(int8_model, test_loader)
            int8_size = converter.get_model_size(int8_model)

            # Calculate speedup and compression
            speedup = fp32_timing['mean_ms'] / int8_timing['mean_ms']
            compression = fp32_size / int8_size

            # Get quality metric name
            quality_key = list(fp32_quality.keys())[0]
            fp32_metric = fp32_quality[quality_key]
            int8_metric = int8_quality[quality_key]
            quality_drop = ((fp32_metric - int8_metric) / fp32_metric) * 100

            # Store results
            result = {
                'model': args.model_type,
                'method': method,
                'fp32_time_ms': fp32_timing['mean_ms'],
                'fp32_std_ms': fp32_timing['std_ms'],
                'int8_time_ms': int8_timing['mean_ms'],
                'int8_std_ms': int8_timing['std_ms'],
                'speedup': speedup,
                'fp32_size_mb': fp32_size,
                'int8_size_mb': int8_size,
                'compression': compression,
                'fp32_quality': fp32_metric,
                'int8_quality': int8_metric,
                'quality_drop_%': quality_drop,
                'metric_name': quality_key
            }

            results.append(result)

            # Print summary
            print("\n[4/4] Results Summary:")
            print(f"  Timing: {fp32_timing['mean_ms']:.2f}ms → {int8_timing['mean_ms']:.2f}ms ({speedup:.2f}x speedup)")
            print(f"  Size: {fp32_size:.2f}MB → {int8_size:.2f}MB ({compression:.2f}x compression)")
            print(f"  Quality: {fp32_metric:.4f} → {int8_metric:.4f} ({quality_drop:.2f}% drop)")

        except Exception as e:
            print(f"[ERROR] Failed to process {method}: {str(e)}")
            continue

    # Create results table
    if results:
        df = pd.DataFrame(results)

        # Save detailed results
        output_dir = 'results/int8_benchmark'
        os.makedirs(output_dir, exist_ok=True)

        df.to_csv(f'{output_dir}/{args.model_type}_int8_results.csv', index=False)

        # Print summary table
        print("\n" + "="*80)
        print(f"{args.model_type.upper()} - INT8 CONVERSION RESULTS")
        print("="*80)

        # Format table for display
        display_df = df[[
            'method', 'fp32_time_ms', 'int8_time_ms', 'speedup',
            'fp32_size_mb', 'int8_size_mb', 'compression',
            'fp32_quality', 'int8_quality', 'quality_drop_%'
        ]].round(2)

        print(tabulate(display_df, headers='keys', tablefmt='grid'))

        # Save summary
        with open(f'{output_dir}/{args.model_type}_summary.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"{args.model_type.upper()} - INT8 CONVERSION RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(tabulate(display_df, headers='keys', tablefmt='grid'))
            f.write("\n\n")

            # Best method analysis
            best_speedup = df.loc[df['speedup'].idxmax()]
            best_quality = df.loc[df['quality_drop_%'].abs().idxmin()]

            f.write("\nBEST METHODS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Best Speedup: {best_speedup['method']} ({best_speedup['speedup']:.2f}x)\n")
            f.write(f"Best Quality: {best_quality['method']} ({best_quality['quality_drop_%']:.2f}% drop)\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real INT8 conversion and CPU benchmark')

    parser.add_argument(
        '--model_type',
        choices=['lstm', 'espcn', 'sasrec'],
        required=True,
        help='Model type to benchmark'
    )

    parser.add_argument(
        '--methods',
        nargs='+',
        default=['none', 'lsq', 'pact', 'adaround', 'apot', 'dsq'],
        help='Quantization methods to test'
    )

    parser.add_argument(
        '--checkpoint_dir',
        default='results/sasrec_fixed',
        help='Directory containing model checkpoints'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )

    parser.add_argument(
        '--num_iterations',
        type=int,
        default=100,
        help='Number of iterations for speed benchmark'
    )

    args = parser.parse_args()

    # Run benchmark
    results = run_full_benchmark(args)

    print("\n[SUCCESS] Benchmark completed!")
    print(f"Results saved to: results/int8_benchmark/{args.model_type}_*")