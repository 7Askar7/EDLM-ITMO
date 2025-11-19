"""
Compare quantization methods (LSQ vs PACT) across all models
"""
import subprocess
import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def run_training(model_type, quantization, args, use_real_data=False):
    """
    Run training for a specific model and quantization method

    Args:
        model_type: 'lstm', 'espcn', or 'sasrec'
        quantization: 'none', 'lsq', or 'pact'
        args: Additional arguments
        use_real_data: Whether to use real data

    Returns:
        Best metric value
    """
    script_map = {
        'lstm': 'train_lstm.py',
        'espcn': 'train_espcn.py',
        'sasrec': 'train_sasrec.py'
    }

    script = script_map[model_type]
    script_path = os.path.join(os.path.dirname(__file__), script)

    # Build command
    cmd = [
        'python', script_path,
        '--quantization', quantization,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--save_dir', os.path.join(args.results_dir, model_type)
    ]

    if use_real_data:
        cmd.append('--use_real_data')

    # Add model-specific arguments
    if model_type == 'lstm':
        cmd.extend([
            '--num_samples', str(args.num_samples),
            '--max_len', str(args.max_len)
        ])
    elif model_type == 'espcn':
        cmd.extend([
            '--num_train', str(args.num_train),
            '--num_test', str(args.num_test)
        ])
    elif model_type == 'sasrec':
        cmd.extend([
            '--num_users', str(args.num_users),
            '--num_items', str(args.num_items),
            '--num_eval_samples', str(args.num_eval_samples)
        ])

    print(f"\nRunning: {' '.join(cmd)}")
    print(f"=" * 80)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        # Parse the best metric from output
        output = result.stdout
        if model_type == 'lstm':
            # Look for "Best Test AUC: X.XXXX"
            for line in output.split('\n'):
                if 'Best Test AUC:' in line:
                    return float(line.split(':')[1].strip())
        elif model_type == 'espcn':
            # Look for "Best Test PSNR: XX.XX dB"
            for line in output.split('\n'):
                if 'Best Test PSNR:' in line:
                    return float(line.split(':')[1].split('dB')[0].strip())
        elif model_type == 'sasrec':
            # Look for "Best Test NDCG@10: X.XXXX"
            for line in output.split('\n'):
                if 'Best Test NDCG' in line:
                    return float(line.split(':')[1].strip())

        return None

    except subprocess.CalledProcessError as e:
        print(f"Error running {model_type} with {quantization}:")
        print(e.stderr)
        return None


def load_history(model_type, quantization, results_dir):
    """Load training history from JSON file"""
    history_file = os.path.join(
        results_dir, model_type,
        f'{model_type}_{quantization}_history.json'
    )

    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return None


def plot_comparison(results, results_dir):
    """
    Plot comparison of quantization methods

    Args:
        results: Dictionary with results
        results_dir: Directory to save plots
    """
    # Create plots directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: Bar chart comparing metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = ['lstm', 'espcn', 'sasrec']
    metrics = ['ROC-AUC', 'PSNR (dB)', 'NDCG@10']

    for idx, (model, metric_name) in enumerate(zip(models, metrics)):
        ax = axes[idx]

        methods = ['Baseline', 'LSQ', 'PACT']
        values = [
            results[model].get('none', 0),
            results[model].get('lsq', 0),
            results[model].get('pact', 0)
        ]

        bars = ax.bar(methods, values, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax.set_ylabel(metric_name)
        ax.set_title(f'{model.upper()} - {metric_name}')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}' if model != 'espcn' else f'{height:.2f}',
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'quantization_comparison.png'), dpi=300)
    plt.close()

    print(f"\nPlots saved to {plots_dir}")


def create_comparison_table(results, results_dir):
    """Create comparison table"""
    # Create DataFrame
    data = []

    for model in ['lstm', 'espcn', 'sasrec']:
        for method in ['none', 'lsq', 'pact']:
            if method in results[model]:
                data.append({
                    'Model': model.upper(),
                    'Method': method.upper(),
                    'Metric': results[model][method]
                })

    df = pd.DataFrame(data)

    # Pivot table
    pivot = df.pivot(index='Model', columns='Method', values='Metric')

    # Calculate relative performance
    pivot['LSQ vs Baseline (%)'] = ((pivot['LSQ'] - pivot['NONE']) / pivot['NONE'] * 100)
    pivot['PACT vs Baseline (%)'] = ((pivot['PACT'] - pivot['NONE']) / pivot['NONE'] * 100)

    # Save to CSV
    csv_path = os.path.join(results_dir, 'comparison_results.csv')
    pivot.to_csv(csv_path)

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(pivot.to_string())
    print(f"\nResults saved to {csv_path}")

    return pivot


def main(args):
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, f'comparison_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)

    print(f"Results will be saved to: {results_dir}")

    # Results dictionary
    results = {
        'lstm': {},
        'espcn': {},
        'sasrec': {}
    }

    # Models to test
    models = []
    if args.test_lstm:
        models.append('lstm')
    if args.test_espcn:
        models.append('espcn')
    if args.test_sasrec:
        models.append('sasrec')

    if not models:
        models = ['lstm', 'espcn', 'sasrec']  # Test all by default

    # Quantization methods
    methods = ['none', 'lsq', 'pact']

    # Run experiments
    for model in models:
        print(f"\n{'=' * 80}")
        print(f"Testing {model.upper()}")
        print(f"{'=' * 80}")

        for method in methods:
            print(f"\nTraining {model.upper()} with {method.upper()}...")

            metric = run_training(model, method, args, args.use_real_data)

            if metric is not None:
                results[model][method] = metric
                print(f"✓ {model.upper()} {method.upper()}: {metric:.4f}")
            else:
                print(f"✗ Failed to train {model.upper()} with {method.upper()}")

    # Create comparison table
    comparison_df = create_comparison_table(results, results_dir)

    # Plot results
    plot_comparison(results, results_dir)

    # Save full results
    results_file = os.path.join(results_dir, 'full_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved to {results_dir}")

    # Determine best method for each model
    print("\n" + "=" * 80)
    print("BEST METHODS")
    print("=" * 80)

    for model in models:
        if results[model]:
            best_method = max(results[model].items(), key=lambda x: x[1])
            print(f"{model.upper()}: {best_method[0].upper()} ({best_method[1]:.4f})")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare quantization methods')

    # General arguments
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--use_real_data', action='store_true',
                        help='Use real datasets (IMDB, BSD, MovieLens)')

    # Model selection
    parser.add_argument('--test_lstm', action='store_true',
                        help='Test LSTM model')
    parser.add_argument('--test_espcn', action='store_true',
                        help='Test ESPCN model')
    parser.add_argument('--test_sasrec', action='store_true',
                        help='Test SASRec model')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for each training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')

    # LSTM-specific
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of synthetic samples for LSTM')
    parser.add_argument('--max_len', type=int, default=128,
                        help='Max sequence length for LSTM')

    # ESPCN-specific
    parser.add_argument('--num_train', type=int, default=1000,
                        help='Number of training samples for ESPCN')
    parser.add_argument('--num_test', type=int, default=200,
                        help='Number of test samples for ESPCN')

    # SASRec-specific
    parser.add_argument('--num_users', type=int, default=1000,
                        help='Number of users for SASRec')
    parser.add_argument('--num_items', type=int, default=500,
                        help='Number of items for SASRec')
    parser.add_argument('--num_eval_samples', type=int, default=500,
                        help='Number of evaluation samples for SASRec')

    args = parser.parse_args()

    main(args)
