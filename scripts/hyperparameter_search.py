"""
Hyperparameter search for quantization methods
Tests different learning rates, batch sizes, and calibration settings
"""
import os
import sys
import json
import argparse
import subprocess
from itertools import product
from datetime import datetime
import pandas as pd
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def run_experiment(model_type, method, lr, batch_size, calibration_batches=None, epochs=30):
    """
    Run a single experiment with specific hyperparameters

    Args:
        model_type: 'lstm', 'espcn', or 'sasrec'
        method: Quantization method
        lr: Learning rate
        batch_size: Batch size
        calibration_batches: Number of calibration batches (for AdaRound, DSQ)
        epochs: Number of epochs

    Returns:
        dict: Results including metrics and hyperparameters
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model_type}_{method}_lr{lr}_bs{batch_size}"
    if calibration_batches:
        exp_name += f"_cal{calibration_batches}"

    # Map model to script
    script_map = {
        'lstm': 'train_lstm_production.py',
        'espcn': 'train_espcn_production.py',
        'sasrec': 'train_sasrec_production.py'
    }

    script = script_map[model_type]
    script_path = os.path.join(os.path.dirname(__file__), script)

    # Build command
    cmd = [
        'python', script_path,
        '--quantization', method,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
        '--seed', '42',
        '--save_dir', f'results/hyperparameter_search/{model_type}',
        '--log_dir', f'logs/hyperparameter_search/{model_type}'
    ]

    # Add calibration batches for methods that need it
    if calibration_batches and method in ['adaround', 'dsq', 'apot']:
        cmd.extend(['--calibration_batches', str(calibration_batches)])

    # Add bit width for quantization methods
    if method != 'none':
        cmd.extend(['--bit_width', '8'])

    print(f"\n[INFO] Running: {exp_name}")
    print(f"Command: {' '.join(cmd)}")

    try:
        # Run the experiment
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        # Load the results
        history_file = f'results/hyperparameter_search/{model_type}/{model_type}_{method}_history.json'

        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)

            # Extract best metrics
            metric_map = {
                'lstm': ('test_auc', 'max'),
                'espcn': ('test_psnr', 'max'),
                'sasrec': ('test_ndcg', 'max')
            }

            metric_key, metric_op = metric_map[model_type]

            if metric_key in history:
                if metric_op == 'max':
                    best_metric = max(history[metric_key])
                else:
                    best_metric = min(history[metric_key])

                return {
                    'model': model_type,
                    'method': method,
                    'lr': lr,
                    'batch_size': batch_size,
                    'calibration_batches': calibration_batches,
                    'epochs': epochs,
                    'best_metric': best_metric,
                    'metric_name': metric_key,
                    'status': 'success'
                }

        return {
            'model': model_type,
            'method': method,
            'lr': lr,
            'batch_size': batch_size,
            'calibration_batches': calibration_batches,
            'epochs': epochs,
            'best_metric': None,
            'status': 'failed'
        }

    except subprocess.TimeoutExpired:
        print(f"[ERROR] Experiment {exp_name} timed out")
        return {
            'model': model_type,
            'method': method,
            'lr': lr,
            'batch_size': batch_size,
            'calibration_batches': calibration_batches,
            'epochs': epochs,
            'best_metric': None,
            'status': 'timeout'
        }
    except Exception as e:
        print(f"[ERROR] Experiment {exp_name} failed: {str(e)}")
        return {
            'model': model_type,
            'method': method,
            'lr': lr,
            'batch_size': batch_size,
            'calibration_batches': calibration_batches,
            'epochs': epochs,
            'best_metric': None,
            'status': 'error'
        }


def hyperparameter_search(args):
    """
    Run hyperparameter search for all specified combinations
    """
    # Define hyperparameter grid
    learning_rates = args.learning_rates
    batch_sizes = args.batch_sizes
    calibration_batches = args.calibration_batches

    # Create results directory
    os.makedirs('results/hyperparameter_search', exist_ok=True)
    os.makedirs('logs/hyperparameter_search', exist_ok=True)

    # Store all results
    all_results = []

    # Run experiments
    for model in args.models:
        print(f"\n{'='*60}")
        print(f"Model: {model.upper()}")
        print(f"{'='*60}")

        for method in args.methods:
            print(f"\n[INFO] Testing method: {method}")

            # Determine if method needs calibration
            needs_calibration = method in ['adaround', 'dsq', 'apot']

            # Create hyperparameter combinations
            if needs_calibration:
                param_combinations = list(product(
                    learning_rates,
                    batch_sizes,
                    calibration_batches
                ))
            else:
                param_combinations = list(product(
                    learning_rates,
                    batch_sizes,
                    [None]
                ))

            # Run experiments for each combination
            for lr, bs, cal in param_combinations:
                result = run_experiment(
                    model_type=model,
                    method=method,
                    lr=lr,
                    batch_size=bs,
                    calibration_batches=cal,
                    epochs=args.epochs
                )
                all_results.append(result)

                # Save intermediate results
                df = pd.DataFrame(all_results)
                df.to_csv('results/hyperparameter_search/intermediate_results.csv', index=False)

    # Create final report
    df = pd.DataFrame(all_results)

    # Save full results
    df.to_csv('results/hyperparameter_search/full_results.csv', index=False)

    # Create summary for each model-method pair
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH SUMMARY")
    print("="*80)

    for model in args.models:
        print(f"\n{model.upper()} Results:")
        print("-"*60)

        model_df = df[df['model'] == model]

        # Find best hyperparameters for each method
        for method in args.methods:
            method_df = model_df[model_df['method'] == method]

            if len(method_df) > 0 and method_df['best_metric'].notna().any():
                best_idx = method_df['best_metric'].idxmax()
                best_row = method_df.loc[best_idx]

                print(f"\n{method.upper()}:")
                print(f"  Best metric: {best_row['best_metric']:.4f}")
                print(f"  Learning rate: {best_row['lr']}")
                print(f"  Batch size: {best_row['batch_size']}")
                if best_row['calibration_batches']:
                    print(f"  Calibration batches: {best_row['calibration_batches']}")

    # Create pivot table for visualization
    for model in args.models:
        model_df = df[df['model'] == model]

        print(f"\n\n{model.upper()} - Hyperparameter Grid:")
        print("="*80)

        for method in args.methods:
            method_df = model_df[model_df['method'] == method]

            if len(method_df) > 0:
                print(f"\n{method.upper()}:")

                # Create pivot table
                pivot = method_df.pivot_table(
                    index='lr',
                    columns='batch_size',
                    values='best_metric',
                    aggfunc='mean'
                )

                if not pivot.empty:
                    print(tabulate(pivot, headers='keys', tablefmt='grid', floatfmt='.4f'))

    # Save summary
    with open('results/hyperparameter_search/SUMMARY.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("HYPERPARAMETER SEARCH SUMMARY\n")
        f.write("="*80 + "\n\n")

        for model in args.models:
            f.write(f"\n{model.upper()} BEST HYPERPARAMETERS:\n")
            f.write("-"*60 + "\n")

            model_df = df[df['model'] == model]

            for method in args.methods:
                method_df = model_df[model_df['method'] == method]

                if len(method_df) > 0 and method_df['best_metric'].notna().any():
                    best_idx = method_df['best_metric'].idxmax()
                    best_row = method_df.loc[best_idx]

                    f.write(f"\n{method.upper()}:\n")
                    f.write(f"  Best metric: {best_row['best_metric']:.4f}\n")
                    f.write(f"  Learning rate: {best_row['lr']}\n")
                    f.write(f"  Batch size: {best_row['batch_size']}\n")
                    if best_row['calibration_batches']:
                        f.write(f"  Calibration batches: {best_row['calibration_batches']}\n")

    print(f"\n[SUCCESS] Results saved to results/hyperparameter_search/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter search for quantization methods')

    parser.add_argument(
        '--models',
        nargs='+',
        default=['lstm', 'espcn', 'sasrec'],
        help='Models to test'
    )

    parser.add_argument(
        '--methods',
        nargs='+',
        default=['lsq', 'pact', 'adaround', 'apot', 'dsq'],
        help='Quantization methods to test'
    )

    parser.add_argument(
        '--learning_rates',
        nargs='+',
        type=float,
        default=[1e-3, 1e-4, 1e-5],
        help='Learning rates to test'
    )

    parser.add_argument(
        '--batch_sizes',
        nargs='+',
        type=int,
        default=[64, 128, 256],
        help='Batch sizes to test'
    )

    parser.add_argument(
        '--calibration_batches',
        nargs='+',
        type=int,
        default=[50, 100, 200],
        help='Calibration batches for AdaRound/DSQ'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of epochs per experiment'
    )

    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Run quick test with minimal hyperparameters'
    )

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        args.models = ['lstm']
        args.methods = ['lsq', 'pact']
        args.learning_rates = [1e-3, 1e-4]
        args.batch_sizes = [128]
        args.calibration_batches = [100]
        args.epochs = 5
        print("[INFO] Running in quick test mode")

    hyperparameter_search(args)