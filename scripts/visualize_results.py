"""
Visualize training results and compare quantization methods
"""
import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style('whitegrid')


def load_history(model_type, quantization, results_dir):
    """Load training history"""
    history_file = os.path.join(
        results_dir, model_type,
        f'{model_type}_{quantization}_history.json'
    )

    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return None


def plot_training_curves(model_type, results_dir, save_dir=None):
    """
    Plot training curves for different quantization methods

    Args:
        model_type: 'lstm', 'espcn', or 'sasrec'
        results_dir: Directory containing results
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    methods = ['none', 'lsq', 'pact']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    # Determine metric name
    metric_names = {
        'lstm': 'ROC-AUC',
        'espcn': 'PSNR (dB)',
        'sasrec': 'NDCG@10'
    }
    metric_key = {
        'lstm': 'test_auc',
        'espcn': 'test_psnr',
        'sasrec': 'test_ndcg'
    }

    metric_name = metric_names.get(model_type, 'Metric')
    metric = metric_key.get(model_type, 'test_metric')

    # Plot training loss
    ax = axes[0]
    for method, color in zip(methods, colors):
        history = load_history(model_type, method, results_dir)
        if history and 'train_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], label=method.upper(), color=color, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title(f'{model_type.upper()} - Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot test loss
    ax = axes[1]
    for method, color in zip(methods, colors):
        history = load_history(model_type, method, results_dir)
        if history and 'test_loss' in history:
            epochs = range(1, len(history['test_loss']) + 1)
            ax.plot(epochs, history['test_loss'], label=method.upper(), color=color, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title(f'{model_type.upper()} - Test Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot test metric
    ax = axes[2]
    for method, color in zip(methods, colors):
        history = load_history(model_type, method, results_dir)
        if history and metric in history:
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], label=method.upper(), color=color, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{model_type.upper()} - {metric_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_type}_training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_all_models(results_dir, save_dir=None):
    """Plot comparison for all models"""
    models = ['lstm', 'espcn', 'sasrec']

    for model in models:
        print(f"\nPlotting {model.upper()}...")
        plot_training_curves(model, results_dir, save_dir)


def create_summary_table(results_dir):
    """Create summary table of best results"""
    models = ['lstm', 'espcn', 'sasrec']
    methods = ['none', 'lsq', 'pact']

    metric_key = {
        'lstm': 'test_auc',
        'espcn': 'test_psnr',
        'sasrec': 'test_ndcg'
    }

    data = []

    for model in models:
        for method in methods:
            history = load_history(model, method, results_dir)
            if history and metric_key[model] in history:
                best_value = max(history[metric_key[model]])
                data.append({
                    'Model': model.upper(),
                    'Method': method.upper(),
                    'Best Metric': best_value
                })

    df = pd.DataFrame(data)

    # Pivot table
    pivot = df.pivot(index='Model', columns='Method', values='Best Metric')

    print("\n" + "="*80)
    print("SUMMARY OF BEST RESULTS")
    print("="*80)
    print(pivot.to_string())

    # Calculate improvements
    if 'NONE' in pivot.columns:
        if 'LSQ' in pivot.columns:
            pivot['LSQ Improvement (%)'] = ((pivot['LSQ'] - pivot['NONE']) / pivot['NONE'] * 100)
        if 'PACT' in pivot.columns:
            pivot['PACT Improvement (%)'] = ((pivot['PACT'] - pivot['NONE']) / pivot['NONE'] * 100)

        print("\n" + "="*80)
        print("IMPROVEMENTS OVER BASELINE")
        print("="*80)
        print(pivot[['LSQ Improvement (%)', 'PACT Improvement (%)']].to_string())

    return pivot


def plot_metric_comparison_bar(results_dir, save_dir=None):
    """Create bar chart comparing all methods across all models"""
    models = ['lstm', 'espcn', 'sasrec']
    methods = ['none', 'lsq', 'pact']

    metric_key = {
        'lstm': 'test_auc',
        'espcn': 'test_psnr',
        'sasrec': 'test_ndcg'
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, model in enumerate(models):
        ax = axes[idx]
        values = []

        for method in methods:
            history = load_history(model, method, results_dir)
            if history and metric_key[model] in history:
                best_value = max(history[metric_key[model]])
                values.append(best_value)
            else:
                values.append(0)

        x = np.arange(len(methods))
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.bar(x, values, color=colors, alpha=0.8)

        ax.set_xlabel('Method')
        ax.set_ylabel('Metric Value')
        ax.set_title(f'{model.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}' if model != 'espcn' else f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'all_models_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def main(args):
    print("Visualizing quantization results...")

    if args.model_type:
        # Plot single model
        plot_training_curves(args.model_type, args.results_dir, args.save_dir)
    else:
        # Plot all models
        plot_all_models(args.results_dir, args.save_dir)

        # Create summary table
        summary = create_summary_table(args.results_dir)

        # Create comparison bar chart
        plot_metric_comparison_bar(args.results_dir, args.save_dir)

        # Save summary
        if args.save_dir:
            summary_path = os.path.join(args.save_dir, 'summary_table.csv')
            summary.to_csv(summary_path)
            print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize quantization results')

    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing results')
    parser.add_argument('--save_dir', type=str, default='./results/plots',
                        help='Directory to save plots')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['lstm', 'espcn', 'sasrec'],
                        help='Specific model to visualize (default: all)')

    args = parser.parse_args()

    main(args)
