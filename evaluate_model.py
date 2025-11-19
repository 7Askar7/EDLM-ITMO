#!/usr/bin/env python
"""
Command-line tool for evaluating SASRec models

Usage:
    python evaluate_model.py --model_path results/sasrec_best.pt --mode both
    python evaluate_model.py --model_path results/sasrec_best.pt --mode 100neg --k 10
    python evaluate_model.py --help
"""

import torch
import argparse
import json
import sys
import os

sys.path.append(os.path.dirname(__file__))

from models.sasrec import SASRec
from utils.datasets import create_synthetic_sasrec_dataset
from utils.data_loader import load_movielens_dataset
from utils.evaluation import (
    evaluate_all_items,
    evaluate_100neg,
    compare_evaluation_modes
)


def load_model(model_path, device='cuda'):
    """Load model from checkpoint"""
    print(f"Loading model from {model_path}...")

    checkpoint = torch.load(model_path, map_location=device)

    # Get model configuration
    if 'args' in checkpoint:
        args = checkpoint['args']
        num_items = args.get('num_items', 3707)
        max_len = args.get('max_len', 200)
        hidden_units = args.get('hidden_units', 128)
        num_blocks = args.get('num_blocks', 2)
        num_heads = args.get('num_heads', 2)
        dropout = args.get('dropout', 0.2)
    else:
        # Use defaults
        print("Warning: Model config not found in checkpoint, using defaults")
        num_items = 3707
        max_len = 200
        hidden_units = 128
        num_blocks = 2
        num_heads = 2
        dropout = 0.2

    # Create model
    model = SASRec(
        num_items=num_items,
        max_seq_len=max_len,
        hidden_units=hidden_units,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Items: {num_items}")
    print(f"  Hidden units: {hidden_units}")
    print(f"  Blocks: {num_blocks}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, num_items


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SASRec model with different protocols',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare both evaluation modes
  python evaluate_model.py --model_path results/sasrec_best.pt --mode both

  # Evaluate with all-items only
  python evaluate_model.py --model_path results/sasrec_best.pt --mode all

  # Evaluate with 100-negative only
  python evaluate_model.py --model_path results/sasrec_best.pt --mode 100neg

  # Use real MovieLens data
  python evaluate_model.py --model_path results/sasrec_best.pt --use_real_data

  # Custom k and negatives
  python evaluate_model.py --model_path results/sasrec_best.pt --k 20 --num_negatives 200
        """
    )

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')

    # Data arguments
    parser.add_argument('--use_real_data', action='store_true',
                        help='Use real MovieLens data instead of synthetic')
    parser.add_argument('--data_dir', type=str, default='./data/ml-1m',
                        help='Path to MovieLens dataset')
    parser.add_argument('--min_rating', type=float, default=4.0,
                        help='Minimum rating threshold')

    # Synthetic data arguments
    parser.add_argument('--num_users', type=int, default=1000,
                        help='Number of synthetic users')
    parser.add_argument('--num_items', type=int, default=500,
                        help='Number of synthetic items')
    parser.add_argument('--avg_seq_len', type=int, default=50,
                        help='Average sequence length for synthetic data')
    parser.add_argument('--max_len', type=int, default=200,
                        help='Maximum sequence length')

    # Evaluation arguments
    parser.add_argument('--mode', type=str, default='both',
                        choices=['all', '100neg', 'both'],
                        help='Evaluation mode: all (all-items), 100neg (negative sampling), both (compare)')
    parser.add_argument('--k', type=int, default=10,
                        help='Top-k for evaluation metrics')
    parser.add_argument('--num_negatives', type=int, default=100,
                        help='Number of negative samples (for 100neg mode)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (None = all)')

    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on')

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'

    # Load model
    model, model_num_items = load_model(args.model_path, args.device)

    # Load data
    print("\nLoading data...")
    if args.use_real_data:
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory not found: {args.data_dir}")
            print("Download MovieLens-1M from: https://grouplens.org/datasets/movielens/1m/")
            return

        train_dataset, test_dataset, num_users, num_items = load_movielens_dataset(
            data_dir=args.data_dir,
            min_rating=args.min_rating
        )
        print(f"Loaded MovieLens: {num_users} users, {num_items} items")
    else:
        print("Using synthetic data...")
        train_dataset, test_dataset, num_users, num_items = create_synthetic_sasrec_dataset(
            num_users=args.num_users,
            num_items=args.num_items,
            avg_seq_len=args.avg_seq_len,
            max_len=args.max_len
        )
        print(f"Generated synthetic data: {num_users} users, {num_items} items")

    print(f"Test set size: {len(test_dataset)}")

    # Check if num_items matches model
    if num_items != model_num_items:
        print(f"\nWarning: Data has {num_items} items but model was trained on {model_num_items} items")
        print("This may cause errors. Proceeding anyway...")

    # Run evaluation
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    results = {}

    if args.mode == 'all':
        print(f"\nEvaluating with ALL-ITEMS mode...")
        all_results = evaluate_all_items(
            model, test_dataset, num_items,
            k=args.k, device=args.device,
            num_samples=args.num_samples,
            verbose=args.verbose
        )
        results['all_items'] = all_results

        print(f"\nResults (All-Items):")
        print(f"  NDCG@{args.k}: {all_results[f'ndcg@{args.k}']:.4f}")
        print(f"  HR@{args.k}: {all_results[f'hr@{args.k}']:.4f}")
        print(f"  Samples: {all_results['num_samples']}")

    elif args.mode == '100neg':
        print(f"\nEvaluating with {args.num_negatives}-NEGATIVE mode...")
        neg_results = evaluate_100neg(
            model, test_dataset, num_items,
            k=args.k, num_negatives=args.num_negatives,
            device=args.device, seed=args.seed,
            num_samples=args.num_samples,
            verbose=args.verbose
        )
        results['neg_sampling'] = neg_results

        print(f"\nResults ({args.num_negatives}-Negative):")
        print(f"  NDCG@{args.k}: {neg_results[f'ndcg@{args.k}']:.4f}")
        print(f"  HR@{args.k}: {neg_results[f'hr@{args.k}']:.4f}")
        print(f"  Samples: {neg_results['num_samples']}")
        print(f"  Seed: {neg_results['seed']}")

    elif args.mode == 'both':
        print(f"\nComparing both evaluation modes...")
        comparison = compare_evaluation_modes(
            model, test_dataset, num_items,
            k=args.k, device=args.device,
            num_negatives=args.num_negatives,
            seed=args.seed,
            num_samples=args.num_samples
        )
        results = comparison

    # Save results
    if args.output:
        print(f"\nSaving results to {args.output}...")
        with open(args.output, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_to_native(obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(x) for x in obj]
                return obj

            json.dump(convert_to_native(results), f, indent=2)
        print(f"Results saved!")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
