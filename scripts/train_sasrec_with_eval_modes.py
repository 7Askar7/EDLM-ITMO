"""
Training script for SASRec with scientifically correct evaluation

This script demonstrates:
1. Using all-items evaluation during training (fast, strict)
2. Using 100-negative evaluation for final reporting (comparable to literature)
3. Statistical comparison between evaluation modes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys
import json
from tqdm import tqdm
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.sasrec import SASRec
from utils.datasets import create_synthetic_sasrec_dataset
from utils.data_loader import load_movielens_dataset
from utils.evaluation import (
    evaluate,
    evaluate_all_items,
    evaluate_100neg,
    compare_evaluation_modes
)


def train_epoch(model, dataloader, optimizer, device, use_amp=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for batch in tqdm(dataloader, desc="Training"):
        log_seqs, pos_seqs, neg_seqs = batch
        log_seqs = log_seqs.to(device)
        pos_seqs = pos_seqs.to(device)
        neg_seqs = neg_seqs.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                pos_logits, neg_logits = model(log_seqs, pos_seqs, neg_seqs)
                # BPR loss
                loss = -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-8).mean()
        else:
            pos_logits, neg_logits = model(log_seqs, pos_seqs, neg_seqs)
            # BPR loss (Bayesian Personalized Ranking)
            loss = -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-8).mean()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    if args.use_real_data and os.path.exists(args.data_dir):
        # Load MovieLens data
        train_dataset, test_dataset, num_users, num_items = load_movielens_dataset(
            data_dir=args.data_dir,
            min_rating=args.min_rating
        )
        print(f"Loaded MovieLens: {num_users} users, {num_items} items")
    else:
        print("Using synthetic data for testing...")
        train_dataset, test_dataset, num_users, num_items = create_synthetic_sasrec_dataset(
            num_users=args.num_users,
            num_items=args.num_items,
            avg_seq_len=args.avg_seq_len,
            max_len=args.max_len
        )

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    print(f"Num users: {num_users}, Num items: {num_items}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create model
    print(f"Creating SASRec model...")
    model = SASRec(
        num_items=num_items,
        max_seq_len=args.max_len,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout
    )
    model = model.to(device)

    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_schedule.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # Training loop
    best_ndcg_all = 0.0
    best_ndcg_100neg = 0.0
    history = {
        'train_loss': [],
        'test_ndcg_all': [],
        'test_hr_all': [],
        'test_ndcg_100neg': [],
        'test_hr_100neg': [],
        'lr': []
    }

    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    print("EVALUATION PROTOCOL:")
    print("- During training: ALL-ITEMS evaluation (strict, fast)")
    print(f"- Final evaluation: 100-NEGATIVE evaluation (standard, comparable)")
    print("=" * 80)

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args.use_amp)

        # Evaluate with ALL-ITEMS (for training monitoring)
        print(f"\nEpoch {epoch + 1}/{args.epochs} - Training evaluation (all-items):")
        all_items_results = evaluate_all_items(
            model, test_dataset, num_items, k=args.k, device=device,
            num_samples=args.num_eval_samples, verbose=True
        )

        test_ndcg_all = all_items_results[f'ndcg@{args.k}']
        test_hr_all = all_items_results[f'hr@{args.k}']

        # Periodically evaluate with 100-NEGATIVE (every N epochs)
        if (epoch + 1) % args.eval_100neg_every == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch + 1}/{args.epochs} - Standard evaluation (100-negative):")
            neg_results = evaluate_100neg(
                model, test_dataset, num_items, k=args.k,
                num_negatives=args.num_negatives, device=device,
                seed=args.seed, num_samples=args.num_eval_samples, verbose=True
            )
            test_ndcg_100neg = neg_results[f'ndcg@{args.k}']
            test_hr_100neg = neg_results[f'hr@{args.k}']
        else:
            # Reuse previous 100-neg scores
            test_ndcg_100neg = history['test_ndcg_100neg'][-1] if history['test_ndcg_100neg'] else 0.0
            test_hr_100neg = history['test_hr_100neg'][-1] if history['test_hr_100neg'] else 0.0

        # Update scheduler based on all-items NDCG
        scheduler.step(test_ndcg_all)

        # Save history
        history['train_loss'].append(train_loss)
        history['test_ndcg_all'].append(test_ndcg_all)
        history['test_hr_all'].append(test_hr_all)
        history['test_ndcg_100neg'].append(test_ndcg_100neg)
        history['test_hr_100neg'].append(test_hr_100neg)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch + 1}/{args.epochs} Summary ({epoch_time:.2f}s):")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  All-Items   -> NDCG@{args.k}: {test_ndcg_all:.4f}, HR@{args.k}: {test_hr_all:.4f}")
        print(f"  100-Negative -> NDCG@{args.k}: {test_ndcg_100neg:.4f}, HR@{args.k}: {test_hr_100neg:.4f}")

        # Save best model based on all-items evaluation
        if test_ndcg_all > best_ndcg_all:
            best_ndcg_all = test_ndcg_all
            save_path = os.path.join(args.save_dir, 'sasrec_best_all.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ndcg_all': best_ndcg_all,
                'args': vars(args)
            }, save_path)
            print(f"  Saved best all-items model: NDCG@{args.k}={best_ndcg_all:.4f}")

        # Save best model based on 100-neg evaluation
        if test_ndcg_100neg > best_ndcg_100neg:
            best_ndcg_100neg = test_ndcg_100neg
            save_path = os.path.join(args.save_dir, 'sasrec_best_100neg.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ndcg_100neg': best_ndcg_100neg,
                'args': vars(args)
            }, save_path)
            print(f"  Saved best 100-neg model: NDCG@{args.k}={best_ndcg_100neg:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s")

    # Final comprehensive evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION - Comparing evaluation modes")
    print("=" * 80)

    comparison_results = compare_evaluation_modes(
        model, test_dataset, num_items, k=args.k,
        device=device, num_negatives=args.num_negatives,
        seed=args.seed, num_samples=None  # Evaluate on full test set
    )

    # Save all results
    results = {
        'best_ndcg_all': best_ndcg_all,
        'best_ndcg_100neg': best_ndcg_100neg,
        'history': history,
        'final_comparison': {
            'all_items': {
                f'ndcg@{args.k}': comparison_results['all_items'][f'ndcg@{args.k}'],
                f'hr@{args.k}': comparison_results['all_items'][f'hr@{args.k}']
            },
            'neg_sampling': {
                f'ndcg@{args.k}': comparison_results['neg_sampling'][f'ndcg@{args.k}'],
                f'hr@{args.k}': comparison_results['neg_sampling'][f'hr@{args.k}']
            },
            'statistics': {
                'ndcg_improvement_pct': 100 * (
                    comparison_results['statistics']['ndcg_ci_neg']['mean'] -
                    comparison_results['statistics']['ndcg_ci_all']['mean']
                ),
                'hr_improvement_pct': 100 * (
                    comparison_results['statistics']['hr_ci_neg']['mean'] -
                    comparison_results['statistics']['hr_ci_all']['mean']
                ),
                'ndcg_significant': comparison_results['statistics']['ndcg_test']['significant'],
                'hr_significant': comparison_results['statistics']['hr_test']['significant']
            },
            'timing': comparison_results['timing']
        },
        'config': vars(args)
    }

    # Save results
    results_path = os.path.join(args.save_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        json.dump(convert_to_native(results), f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Best All-Items NDCG@{args.k}: {best_ndcg_all:.4f}")
    print(f"Best 100-Neg NDCG@{args.k}: {best_ndcg_100neg:.4f}")
    print(f"\nFinal evaluation (full test set):")
    print(f"  All-Items:    NDCG@{args.k}={comparison_results['all_items'][f'ndcg@{args.k}']:.4f}, "
          f"HR@{args.k}={comparison_results['all_items'][f'hr@{args.k}']:.4f}")
    print(f"  100-Negative: NDCG@{args.k}={comparison_results['neg_sampling'][f'ndcg@{args.k}']:.4f}, "
          f"HR@{args.k}={comparison_results['neg_sampling'][f'hr@{args.k}']:.4f}")
    print("\nFor paper reporting, use the 100-Negative scores!")
    print("=" * 80)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train SASRec with scientifically correct evaluation'
    )

    # Data arguments
    parser.add_argument('--use_real_data', action='store_true',
                        help='Use real MovieLens data')
    parser.add_argument('--data_dir', type=str, default='./data/ml-1m',
                        help='Path to MovieLens dataset')
    parser.add_argument('--min_rating', type=float, default=4.0,
                        help='Minimum rating threshold')
    parser.add_argument('--num_users', type=int, default=1000,
                        help='Number of synthetic users')
    parser.add_argument('--num_items', type=int, default=500,
                        help='Number of synthetic items')
    parser.add_argument('--avg_seq_len', type=int, default=50,
                        help='Average sequence length for synthetic data')

    # Model arguments
    parser.add_argument('--max_len', type=int, default=200,
                        help='Maximum sequence length')
    parser.add_argument('--hidden_units', type=int, default=128,
                        help='Hidden units dimension')
    parser.add_argument('--num_blocks', type=int, default=2,
                        help='Number of self-attention blocks')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')

    # Evaluation arguments
    parser.add_argument('--k', type=int, default=10,
                        help='Top-k for evaluation')
    parser.add_argument('--num_eval_samples', type=int, default=1000,
                        help='Number of samples for quick evaluation during training')
    parser.add_argument('--num_negatives', type=int, default=100,
                        help='Number of negative samples for 100-neg protocol')
    parser.add_argument('--eval_100neg_every', type=int, default=5,
                        help='Evaluate with 100-neg protocol every N epochs')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='./results/sasrec_eval',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)
