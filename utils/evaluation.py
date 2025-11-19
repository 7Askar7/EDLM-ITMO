"""Scientific evaluation protocols for recommendation systems.

This module implements rigorous evaluation protocols for sequential
recommendation models, following best practices from academic literature.

Evaluation Modes:
    1. ALL-ITEMS: Ranks all items in catalog (strict, for internal monitoring)
    2. 100-NEGATIVE: Standard sampling protocol (for fair comparison with
        published baselines)

The 100-negative sampling protocol is standard in recommendation systems
research and allows for fair comparison with published baselines while being
computationally more efficient than all-items ranking.

References:
    - "Neural Collaborative Filtering" (He et al., WWW 2017)
    - "BPR: Bayesian Personalized Ranking" (Rendle et al., UAI 2009)
    - "Self-Attentive Sequential Recommendation" (Kang & McAuley, ICDM 2018)

Key Functions:
    - evaluate_all_items: Strict evaluation ranking all items
    - evaluate_100neg: Standard negative sampling protocol
    - compare_evaluation_modes: Statistical comparison of both modes
"""
import time
from collections import defaultdict

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm


def calculate_ndcg_at_k(predictions, target_items, k=10, item_offset=1):
    """Calculate NDCG@k (Normalized Discounted Cumulative Gain).

    NDCG is a ranking quality metric that gives higher weight to relevant
    items appearing at the top of the ranking. The gain is accumulated from
    top to bottom with a logarithmic reduction factor.

    Args:
        predictions: Predicted item scores (numpy array).
        target_items: Ground truth items (set or list).
        k: Number of top items to consider (default: 10).
        item_offset: Offset to convert array indices to item IDs (default: 1).

    Returns:
        NDCG@k score (float) in range [0, 1], where 1 is perfect ranking.

    Note:
        Uses the standard NDCG formula with log2(i+1) discounting factor.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    # Get top-k predictions
    if len(predictions) > k:
        top_k_indices = np.argsort(predictions)[::-1][:k]
    else:
        top_k_indices = np.argsort(predictions)[::-1]

    # Create relevance scores (1 if in ground truth, 0 otherwise)
    relevance_scores = []
    for idx in top_k_indices:
        item_id = idx + item_offset
        if item_id in target_items:
            relevance_scores.append(1.0)
        else:
            relevance_scores.append(0.0)

    if len(relevance_scores) == 0 or sum(relevance_scores) == 0:
        return 0.0

    relevance_scores = np.array(relevance_scores)

    # Calculate DCG
    dcg = relevance_scores[0]
    for i in range(1, len(relevance_scores)):
        dcg += relevance_scores[i] / np.log2(i + 1)

    # Calculate IDCG (ideal DCG)
    ideal_scores = np.sort(relevance_scores)[::-1]
    idcg = ideal_scores[0]
    for i in range(1, len(ideal_scores)):
        idcg += ideal_scores[i] / np.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_hit_rate(predictions, target_items, k=10, item_offset=1):
    """
    Calculate Hit Rate@k (HR@k)

    Args:
        predictions: Predicted item scores
        target_items: Ground truth items
        k: Number of top items to consider
        item_offset: Offset to convert array indices to item IDs

    Returns:
        1 if any ground truth item is in top-k, 0 otherwise
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    # Get top-k predictions
    if len(predictions) > k:
        top_k_indices = np.argsort(predictions)[::-1][:k]
    else:
        top_k_indices = np.argsort(predictions)[::-1]

    # Check if any ground truth item is in top-k
    for idx in top_k_indices:
        item_id = idx + item_offset
        if item_id in target_items:
            return 1.0

    return 0.0


def sample_negatives(target_item, user_history, num_items, num_negatives=100, seed=None):
    """
    Sample negative items for evaluation

    Args:
        target_item: Target item ID
        user_history: Set of items in user's history
        num_items: Total number of items
        num_negatives: Number of negative samples
        seed: Random seed for reproducibility

    Returns:
        List of negative item IDs
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    negatives = []
    max_attempts = num_negatives * 10  # Prevent infinite loop
    attempts = 0

    while len(negatives) < num_negatives and attempts < max_attempts:
        # Sample random item (1 to num_items)
        neg_item = rng.randint(1, num_items + 1)

        # Check if it's not in history and not the target
        if neg_item not in user_history and neg_item != target_item:
            negatives.append(neg_item)

        attempts += 1

    if len(negatives) < num_negatives:
        print(f"Warning: Could only sample {len(negatives)} negatives (requested {num_negatives})")

    return negatives


def evaluate_all_items(model, dataset, num_items, k=10, device='cuda',
                       num_samples=None, verbose=True):
    """
    Evaluate model by ranking ALL items (strict evaluation)

    This is the STRICT evaluation mode where the model must rank the target
    item among ALL 3000+ items in the catalog. This is harder than the
    100-negative sampling protocol but faster to compute.

    Args:
        model: SASRec model
        dataset: Dataset object with test data
        num_items: Total number of items in catalog
        k: Top-k for evaluation metrics
        device: Device to run on
        num_samples: Number of test samples to evaluate (None = all)
        verbose: Whether to show progress bar

    Returns:
        Dictionary with metrics: {
            'ndcg@k': float,
            'hr@k': float,
            'ndcg_scores': list,
            'hr_scores': list,
            'num_samples': int
        }
    """
    model.eval()

    ndcg_scores = []
    hr_scores = []

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Pre-compute all item embeddings for efficiency
    all_items = torch.arange(1, num_items + 1).to(device)

    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Evaluating (All-Items)") if verbose else dataloader

        for batch_idx, batch in enumerate(iterator):
            if num_samples and batch_idx * 128 >= num_samples:
                break

            log_seqs, pos_seqs, _ = batch
            log_seqs = log_seqs.to(device)

            for i in range(log_seqs.size(0)):
                seq = log_seqs[i:i+1]  # (1, seq_len)

                # Get the target item (last non-zero item in pos_seqs)
                target_items = pos_seqs[i]
                target_items = target_items[target_items != 0].tolist()

                if len(target_items) == 0:
                    continue

                target = target_items[-1]  # Last item is the target

                # Get predictions for ALL items
                predictions = model.predict(seq, all_items)
                predictions = predictions.squeeze(0).cpu().numpy()

                # Calculate metrics
                ndcg = calculate_ndcg_at_k(predictions, [target], k)
                hr = calculate_hit_rate(predictions, [target], k)

                ndcg_scores.append(ndcg)
                hr_scores.append(hr)

    return {
        f'ndcg@{k}': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        f'hr@{k}': np.mean(hr_scores) if hr_scores else 0.0,
        'ndcg_scores': ndcg_scores,
        'hr_scores': hr_scores,
        'num_samples': len(ndcg_scores)
    }


def evaluate_100neg(model, dataset, num_items, k=10, num_negatives=100,
                    device='cuda', seed=42, num_samples=None, verbose=True):
    """
    Evaluate model using 100-negative sampling protocol (standard in literature)

    This is the STANDARD evaluation protocol used in most papers:
    - For each test case: 1 target item + 100 sampled negative items
    - Rank these 101 items
    - Calculate NDCG@k and HR@k

    This is EASIER than all-items ranking but COMPARABLE to published baselines.

    Reference:
    - "Neural Collaborative Filtering" (He et al., WWW 2017)
    - "BPR: Bayesian Personalized Ranking" (Rendle et al., UAI 2009)

    Args:
        model: SASRec model
        dataset: Dataset object with test data
        num_items: Total number of items in catalog
        k: Top-k for evaluation metrics
        num_negatives: Number of negative samples (default: 100)
        device: Device to run on
        seed: Random seed for deterministic sampling
        num_samples: Number of test samples to evaluate (None = all)
        verbose: Whether to show progress bar

    Returns:
        Dictionary with metrics: {
            'ndcg@k': float,
            'hr@k': float,
            'ndcg_scores': list,
            'hr_scores': list,
            'num_samples': int,
            'seed': int,
            'num_negatives': int
        }
    """
    model.eval()

    ndcg_scores = []
    hr_scores = []

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Build user histories for negative sampling
    user_histories = defaultdict(set)
    for batch_idx, batch in enumerate(dataloader):
        log_seqs, pos_seqs, _ = batch

        for i in range(log_seqs.size(0)):
            # Get all items in this user's history
            seq_items = log_seqs[i][log_seqs[i] != 0].tolist()
            pos_items = pos_seqs[i][pos_seqs[i] != 0].tolist()

            user_idx = batch_idx * 128 + i
            user_histories[user_idx].update(seq_items)
            user_histories[user_idx].update(pos_items)

    # Evaluation with negative sampling
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    with torch.no_grad():
        iterator = tqdm(dataloader, desc=f"Evaluating (100-Neg)") if verbose else dataloader

        for batch_idx, batch in enumerate(iterator):
            if num_samples and batch_idx * 128 >= num_samples:
                break

            log_seqs, pos_seqs, _ = batch
            log_seqs = log_seqs.to(device)

            for i in range(log_seqs.size(0)):
                seq = log_seqs[i:i+1]  # (1, seq_len)

                # Get the target item
                target_items = pos_seqs[i]
                target_items = target_items[target_items != 0].tolist()

                if len(target_items) == 0:
                    continue

                target = target_items[-1]
                user_idx = batch_idx * 128 + i
                user_history = user_histories[user_idx]

                # Sample negative items (deterministic with seed + user_idx)
                negatives = sample_negatives(
                    target,
                    user_history,
                    num_items,
                    num_negatives,
                    seed=seed + user_idx  # Deterministic per user
                )

                # Create candidate set: target + negatives
                candidates = [target] + negatives
                candidate_tensor = torch.LongTensor(candidates).to(device)

                # Get predictions for candidate items only
                predictions = model.predict(seq, candidate_tensor)
                predictions = predictions.squeeze(0).cpu().numpy()

                # Calculate metrics using candidate indices (target is at index 0)
                ndcg = calculate_ndcg_at_k(predictions, [0], k, item_offset=0)
                hr = calculate_hit_rate(predictions, [0], k, item_offset=0)

                ndcg_scores.append(ndcg)
                hr_scores.append(hr)

    return {
        f'ndcg@{k}': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        f'hr@{k}': np.mean(hr_scores) if hr_scores else 0.0,
        'ndcg_scores': ndcg_scores,
        'hr_scores': hr_scores,
        'num_samples': len(ndcg_scores),
        'seed': seed,
        'num_negatives': num_negatives
    }


def compute_confidence_interval(scores, confidence=0.95):
    """
    Compute confidence interval for evaluation metrics

    Args:
        scores: List of metric scores
        confidence: Confidence level (default: 0.95)

    Returns:
        Dictionary with mean, std, ci_lower, ci_upper
    """
    if len(scores) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'n': 0
        }

    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    n = len(scores)

    # Calculate confidence interval using t-distribution
    se = std / np.sqrt(n)
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * se

    return {
        'mean': mean,
        'std': std,
        'se': se,
        'ci_lower': mean - margin,
        'ci_upper': mean + margin,
        'n': n
    }


def paired_t_test(scores1, scores2):
    """
    Perform paired t-test between two sets of scores

    Args:
        scores1: First set of scores
        scores2: Second set of scores

    Returns:
        Dictionary with t_statistic, p_value, effect_size (Cohen's d)
    """
    if len(scores1) != len(scores2):
        raise ValueError("Score lists must have same length for paired t-test")

    if len(scores1) == 0:
        return {
            't_statistic': 0.0,
            'p_value': 1.0,
            'effect_size': 0.0,
            'significant': False
        }

    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)

    # Cohen's d for paired samples
    diff = scores1 - scores2
    effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'significant': p_value < 0.05
    }


def compare_evaluation_modes(model, dataset, num_items, k=10, device='cuda',
                             num_negatives=100, seed=42, num_samples=None):
    """
    Compare all-items vs 100-negative evaluation modes

    This function runs both evaluation protocols and provides statistical
    comparison to understand the difference in difficulty.

    Args:
        model: SASRec model
        dataset: Dataset object with test data
        num_items: Total number of items
        k: Top-k for evaluation
        device: Device to run on
        num_negatives: Number of negatives for sampling mode
        seed: Random seed for reproducibility
        num_samples: Number of samples to evaluate (None = all)

    Returns:
        Dictionary with comparison results
    """
    print("=" * 80)
    print("EVALUATION MODE COMPARISON")
    print("=" * 80)
    print(f"Dataset: {len(dataset)} test cases")
    print(f"Total items: {num_items}")
    print(f"Evaluation metric: NDCG@{k}, HR@{k}")
    print(f"Seed: {seed}")
    print("=" * 80)

    # Run all-items evaluation
    print("\n1. ALL-ITEMS EVALUATION (Strict)")
    print("   - Ranking ALL items in catalog")
    print("   - Use for: internal monitoring, training progress")
    start_time = time.time()
    all_items_results = evaluate_all_items(
        model, dataset, num_items, k, device, num_samples, verbose=True
    )
    all_items_time = time.time() - start_time

    print(f"\n   Results:")
    print(f"   NDCG@{k}: {all_items_results[f'ndcg@{k}']:.4f}")
    print(f"   HR@{k}: {all_items_results[f'hr@{k}']:.4f}")
    print(f"   Time: {all_items_time:.2f}s")

    # Run 100-negative evaluation
    print(f"\n2. {num_negatives}-NEGATIVE EVALUATION (Standard)")
    print(f"   - Ranking 1 target + {num_negatives} sampled negatives")
    print("   - Use for: final reporting, comparison with literature")
    start_time = time.time()
    neg_sampling_results = evaluate_100neg(
        model, dataset, num_items, k, num_negatives, device, seed, num_samples, verbose=True
    )
    neg_sampling_time = time.time() - start_time

    print(f"\n   Results:")
    print(f"   NDCG@{k}: {neg_sampling_results[f'ndcg@{k}']:.4f}")
    print(f"   HR@{k}: {neg_sampling_results[f'hr@{k}']:.4f}")
    print(f"   Time: {neg_sampling_time:.2f}s")

    # Statistical comparison
    print("\n3. STATISTICAL COMPARISON")
    print("=" * 80)

    # NDCG comparison
    ndcg_ci_all = compute_confidence_interval(all_items_results['ndcg_scores'])
    ndcg_ci_neg = compute_confidence_interval(neg_sampling_results['ndcg_scores'])

    print(f"\nNDCG@{k}:")
    print(f"  All-Items:   {ndcg_ci_all['mean']:.4f} ± {ndcg_ci_all['std']:.4f}")
    print(f"               95% CI: [{ndcg_ci_all['ci_lower']:.4f}, {ndcg_ci_all['ci_upper']:.4f}]")
    print(f"  100-Negative: {ndcg_ci_neg['mean']:.4f} ± {ndcg_ci_neg['std']:.4f}")
    print(f"               95% CI: [{ndcg_ci_neg['ci_lower']:.4f}, {ndcg_ci_neg['ci_upper']:.4f}]")

    ndcg_test = paired_t_test(
        neg_sampling_results['ndcg_scores'],
        all_items_results['ndcg_scores']
    )
    print(f"\n  Paired t-test:")
    print(f"    t-statistic: {ndcg_test['t_statistic']:.4f}")
    print(f"    p-value: {ndcg_test['p_value']:.4e}")
    print(f"    Effect size (Cohen's d): {ndcg_test['effect_size']:.4f}")
    print(f"    Significant: {ndcg_test['significant']}")

    # HR comparison
    hr_ci_all = compute_confidence_interval(all_items_results['hr_scores'])
    hr_ci_neg = compute_confidence_interval(neg_sampling_results['hr_scores'])

    print(f"\nHR@{k}:")
    print(f"  All-Items:    {hr_ci_all['mean']:.4f} ± {hr_ci_all['std']:.4f}")
    print(f"               95% CI: [{hr_ci_all['ci_lower']:.4f}, {hr_ci_all['ci_upper']:.4f}]")
    print(f"  100-Negative: {hr_ci_neg['mean']:.4f} ± {hr_ci_neg['std']:.4f}")
    print(f"               95% CI: [{hr_ci_neg['ci_lower']:.4f}, {hr_ci_neg['ci_upper']:.4f}]")

    hr_test = paired_t_test(
        neg_sampling_results['hr_scores'],
        all_items_results['hr_scores']
    )
    print(f"\n  Paired t-test:")
    print(f"    t-statistic: {hr_test['t_statistic']:.4f}")
    print(f"    p-value: {hr_test['p_value']:.4e}")
    print(f"    Effect size (Cohen's d): {hr_test['effect_size']:.4f}")
    print(f"    Significant: {hr_test['significant']}")

    # Summary
    print("\n4. SUMMARY")
    print("=" * 80)
    print(f"The 100-negative protocol typically shows {num_negatives}/{num_items} = "
          f"{100*num_negatives/num_items:.1f}% of the difficulty of all-items ranking.")
    print(f"\nNDCG improvement: {100*(ndcg_ci_neg['mean'] - ndcg_ci_all['mean']):.1f}%")
    print(f"HR improvement: {100*(hr_ci_neg['mean'] - hr_ci_all['mean']):.1f}%")
    print(f"\nSpeedup: {all_items_time / neg_sampling_time:.2f}x")

    print("\nRECOMMENDATIONS:")
    print("- Use ALL-ITEMS during training for strict monitoring")
    print("- Use 100-NEGATIVE for final evaluation and paper comparisons")
    print("- Always report which protocol was used in papers/reports")
    print("=" * 80)

    return {
        'all_items': all_items_results,
        'neg_sampling': neg_sampling_results,
        'statistics': {
            'ndcg_ci_all': ndcg_ci_all,
            'ndcg_ci_neg': ndcg_ci_neg,
            'hr_ci_all': hr_ci_all,
            'hr_ci_neg': hr_ci_neg,
            'ndcg_test': ndcg_test,
            'hr_test': hr_test
        },
        'timing': {
            'all_items_time': all_items_time,
            'neg_sampling_time': neg_sampling_time,
            'speedup': all_items_time / neg_sampling_time
        }
    }


def evaluate(model, dataset, num_items, k=10, device='cuda',
            mode='all', num_negatives=100, seed=42, num_samples=None, verbose=True):
    """
    Unified evaluation function supporting multiple modes

    Args:
        model: SASRec model
        dataset: Dataset object with test data
        num_items: Total number of items
        k: Top-k for evaluation
        device: Device to run on
        mode: Evaluation mode - 'all', '100neg', or 'both'
        num_negatives: Number of negatives (for mode='100neg' or 'both')
        seed: Random seed for reproducibility
        num_samples: Number of samples to evaluate (None = all)
        verbose: Whether to show progress

    Returns:
        Dictionary with evaluation results
    """
    if mode == 'all':
        return evaluate_all_items(model, dataset, num_items, k, device, num_samples, verbose)
    elif mode == '100neg':
        return evaluate_100neg(model, dataset, num_items, k, num_negatives, device, seed, num_samples, verbose)
    elif mode == 'both':
        return compare_evaluation_modes(model, dataset, num_items, k, device, num_negatives, seed, num_samples)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'all', '100neg', or 'both'")
