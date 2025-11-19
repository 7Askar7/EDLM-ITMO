"""Evaluation metrics for different machine learning models.

This module provides metrics for evaluating:
- LSTM text classification: ROC-AUC
- ESPCN super-resolution: PSNR (Peak Signal-to-Noise Ratio)
- SASRec recommendation: NDCG@k, Hit Rate@k, Recall@k, Precision@k

Also includes a MetricsTracker class for tracking metrics during training
and evaluation functions for each model type.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def calculate_roc_auc(y_true, y_pred):
    """Calculate ROC-AUC score for binary classification.

    Computes the Area Under the Receiver Operating Characteristic Curve
    from prediction scores. Handles both PyTorch tensors and NumPy arrays.

    Args:
        y_true: True binary labels (0 or 1). Can be numpy array or tensor.
        y_pred: Predicted probabilities for the positive class.
            Can be numpy array or tensor.

    Returns:
        ROC-AUC score (float) in range [0, 1]. Returns 0.0 if calculation
        fails (e.g., only one class present in y_true).

    Raises:
        ValueError: If inputs have incompatible shapes (printed as warning).
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Flatten arrays to 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    try:
        score = roc_auc_score(y_true, y_pred)
    except ValueError as e:
        print(f"Error calculating ROC-AUC: {e}")
        score = 0.0

    return score


def calculate_psnr(img1, img2, max_value=1.0):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

    PSNR is a quality metric for image reconstruction, commonly used in
    super-resolution and compression tasks. Higher values indicate better
    quality (less distortion).

    Args:
        img1: First image (tensor or numpy array).
        img2: Second image (tensor or numpy array), must have same shape
            as img1.
        max_value: Maximum possible pixel value (default: 1.0 for
            normalized images in [0, 1]).

    Returns:
        PSNR value in decibels (dB). Returns infinity if images are
        identical (MSE = 0).

    Note:
        Typical PSNR values:
        - 30-50 dB: Good quality
        - 20-30 dB: Acceptable quality
        - <20 dB: Poor quality
    """
    # Convert numpy arrays to tensors if needed
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)

    # Calculate Mean Squared Error
    mse = torch.mean((img1 - img2) ** 2)

    # Handle perfect reconstruction case
    if mse == 0:
        return float('inf')

    # Calculate PSNR: 20 * log10(MAX / sqrt(MSE))
    psnr = 20 * torch.log10(torch.tensor(max_value) / torch.sqrt(mse))
    return psnr.item()


def calculate_psnr_batch(img1_batch, img2_batch, max_value=1.0):
    """
    Calculate average PSNR for a batch of images

    Args:
        img1_batch: Batch of first images (batch, channels, height, width)
        img2_batch: Batch of second images
        max_value: Maximum pixel value

    Returns:
        Average PSNR across batch
    """
    batch_size = img1_batch.size(0)
    psnr_values = []

    for i in range(batch_size):
        psnr = calculate_psnr(img1_batch[i], img2_batch[i], max_value)
        if psnr != float('inf'):
            psnr_values.append(psnr)

    if len(psnr_values) == 0:
        return 0.0

    return np.mean(psnr_values)


def calculate_ndcg(relevance_scores, k=10):
    """
    Calculate NDCG@k (Normalized Discounted Cumulative Gain)

    Args:
        relevance_scores: List of relevance scores in ranked order
        k: Number of top items to consider

    Returns:
        NDCG@k score
    """
    if len(relevance_scores) == 0:
        return 0.0

    relevance_scores = np.array(relevance_scores)[:k]

    # Calculate DCG
    # FIXED: Standard NDCG formula uses log2(i+1) not log2(i+2)
    # Position 1 (index 0): no discount
    # Position i (index i-1): discount by log2(i)
    dcg = relevance_scores[0]
    for i in range(1, len(relevance_scores)):
        dcg += relevance_scores[i] / np.log2(i + 1)  # FIXED: i+1 not i+2

    # Calculate IDCG (ideal DCG)
    ideal_scores = np.sort(relevance_scores)[::-1]  # Sort in descending order
    idcg = ideal_scores[0]
    for i in range(1, len(ideal_scores)):
        idcg += ideal_scores[i] / np.log2(i + 1)  # FIXED: i+1 not i+2

    if idcg == 0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


def calculate_ndcg_at_k(predictions, ground_truth, k=10, item_offset=1,
                        item_ids=None):
    """
    Calculate NDCG@k for recommendation systems

    Args:
        predictions: Predicted item scores (numpy array or list)
        ground_truth: Ground truth items (set or list)
        k: Number of top items to consider
        item_offset: Offset to convert array indices to item IDs (default: 1)
                     If predictions[0] corresponds to item_id=1, use offset=1
        item_ids: Optional array/list mapping prediction indices to concrete item IDs.
                  When provided, item_offset is ignored.

    Returns:
        NDCG@k score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    if item_ids is not None:
        item_ids = np.array(item_ids)

    # Get top-k predictions
    if len(predictions) > k:
        top_k_indices = np.argsort(predictions)[::-1][:k]
    else:
        top_k_indices = np.argsort(predictions)[::-1]

    # Create relevance scores (1 if in ground truth, 0 otherwise)
    relevance_scores = []
    for idx in top_k_indices:
        # Convert array index to item_id
        if item_ids is not None:
            item_id = item_ids[idx]
        else:
            item_id = idx + item_offset
        if item_id in ground_truth:
            relevance_scores.append(1.0)
        else:
            relevance_scores.append(0.0)

    return calculate_ndcg(relevance_scores, k)


def calculate_hit_rate(predictions, ground_truth, k=10, item_offset=1,
                       item_ids=None):
    """
    Calculate Hit Rate@k (HR@k)

    Args:
        predictions: Predicted item scores
        ground_truth: Ground truth items
        k: Number of top items to consider
        item_offset: Offset to convert array indices to item IDs (default: 1)
                     If predictions[0] corresponds to item_id=1, use offset=1
        item_ids: Optional mapping of prediction indices to actual item IDs

    Returns:
        1 if any ground truth item is in top-k, 0 otherwise
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    if item_ids is not None:
        item_ids = np.array(item_ids)

    # Get top-k predictions
    if len(predictions) > k:
        top_k_indices = np.argsort(predictions)[::-1][:k]
    else:
        top_k_indices = np.argsort(predictions)[::-1]

    # Check if any ground truth item is in top-k
    for idx in top_k_indices:
        # Convert array index to item_id
        if item_ids is not None:
            item_id = item_ids[idx]
        else:
            item_id = idx + item_offset
        if item_id in ground_truth:
            return 1.0

    return 0.0


def calculate_recall_at_k(predictions, ground_truth, k=10):
    """
    Calculate Recall@k

    Args:
        predictions: Predicted item scores
        ground_truth: Ground truth items (set or list)
        k: Number of top items to consider

    Returns:
        Recall@k score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    if len(ground_truth) == 0:
        return 0.0

    # Get top-k predictions
    if len(predictions) > k:
        top_k_indices = set(np.argsort(predictions)[::-1][:k])
    else:
        top_k_indices = set(np.argsort(predictions)[::-1])

    # Calculate recall
    ground_truth_set = set(ground_truth)
    hits = len(top_k_indices & ground_truth_set)
    recall = hits / len(ground_truth_set)

    return recall


def calculate_precision_at_k(predictions, ground_truth, k=10):
    """
    Calculate Precision@k

    Args:
        predictions: Predicted item scores
        ground_truth: Ground truth items (set or list)
        k: Number of top items to consider

    Returns:
        Precision@k score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    # Get top-k predictions
    if len(predictions) > k:
        top_k_indices = set(np.argsort(predictions)[::-1][:k])
    else:
        top_k_indices = set(np.argsort(predictions)[::-1])
        k = len(top_k_indices)

    if k == 0:
        return 0.0

    # Calculate precision
    ground_truth_set = set(ground_truth)
    hits = len(top_k_indices & ground_truth_set)
    precision = hits / k

    return precision


class MetricsTracker:
    """Track multiple metrics during training and evaluation.

    Accumulates metric values and provides convenient methods for
    computing statistics like averages. Useful for monitoring training
    progress across epochs and batches.

    Attributes:
        metric_names: List of metric names to track.
        metrics: Dictionary mapping metric names to lists of values.

    Example:
        >>> tracker = MetricsTracker(['loss', 'accuracy'])
        >>> tracker.update('loss', 0.5)
        >>> tracker.update('accuracy', 0.92)
        >>> print(tracker.get_average('loss'))
        0.5
    """

    def __init__(self, metric_names):
        """Initialize the metrics tracker.

        Args:
            metric_names: List of metric names to track (e.g., ['loss',
                'accuracy']).
        """
        self.metric_names = metric_names
        self.reset()

    def reset(self):
        """Reset all metrics to empty lists."""
        self.metrics = {name: [] for name in self.metric_names}

    def update(self, metric_name, value):
        """Update a metric by appending a new value.

        Args:
            metric_name: Name of the metric to update.
            value: Value to append to the metric's history.

        Raises:
            ValueError: If metric_name is not in the tracked metrics.
        """
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def get_average(self, metric_name):
        """Get the average value of a metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            Average value (float). Returns 0.0 if no values recorded.

        Raises:
            ValueError: If metric_name is not in the tracked metrics.
        """
        if metric_name not in self.metrics:
            raise ValueError(f"Unknown metric: {metric_name}")

        values = self.metrics[metric_name]
        if len(values) == 0:
            return 0.0

        return np.mean(values)

    def get_all_averages(self):
        """Get average values for all tracked metrics.

        Returns:
            Dictionary mapping metric names to their average values.
        """
        return {name: self.get_average(name) for name in self.metric_names}

    def __str__(self):
        """Return string representation of average metrics.

        Returns:
            Formatted string with all metrics and their averages.
        """
        avg_metrics = self.get_all_averages()
        return ", ".join(
            [f"{name}: {value:.4f}" for name, value in avg_metrics.items()]
        )


def evaluate_lstm_classifier(model, dataloader, device='cuda'):
    """
    Evaluate LSTM classifier

    Args:
        model: LSTM model
        dataloader: Data loader
        device: Device to run on

    Returns:
        Dictionary with metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                texts, lengths, labels = batch
                texts, labels = texts.to(device), labels.to(device)
                outputs = model.predict_proba(texts, lengths)
            else:
                texts, labels = batch
                texts, labels = texts.to(device), labels.to(device)
                outputs = model.predict_proba(texts)

            all_predictions.append(outputs.cpu())
            all_labels.append(labels.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    roc_auc = calculate_roc_auc(all_labels, all_predictions)

    return {
        'roc_auc': roc_auc
    }


def evaluate_espcn(model, dataloader, device='cuda'):
    """
    Evaluate ESPCN model

    Args:
        model: ESPCN model
        dataloader: Data loader
        device: Device to run on

    Returns:
        Dictionary with metrics
    """
    model.eval()
    psnr_values = []

    with torch.no_grad():
        for batch in dataloader:
            lr_images, hr_images = batch
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            outputs = model(lr_images)

            # Calculate PSNR for each image in batch
            batch_psnr = calculate_psnr_batch(outputs, hr_images)
            psnr_values.append(batch_psnr)

    avg_psnr = np.mean(psnr_values)

    return {
        'psnr': avg_psnr
    }


def evaluate_sasrec(model, dataset, k=10, device='cuda'):
    """
    Evaluate SASRec model

    Args:
        model: SASRec model
        dataset: Dataset object with test data
        k: Top-k for evaluation
        device: Device to run on

    Returns:
        Dictionary with metrics
    """
    model.eval()

    ndcg_scores = []
    hr_scores = []

    with torch.no_grad():
        for user_seq, target_items in dataset:
            user_seq = torch.LongTensor([user_seq]).to(device)

            # Get predictions for all items
            all_items = torch.arange(1, model.num_items + 1).to(device)
            predictions = model.predict(user_seq, all_items)
            predictions = predictions.squeeze(0)

            # Calculate metrics
            ndcg = calculate_ndcg_at_k(predictions, target_items, k)
            hr = calculate_hit_rate(predictions, target_items, k)

            ndcg_scores.append(ndcg)
            hr_scores.append(hr)

    avg_ndcg = np.mean(ndcg_scores)
    avg_hr = np.mean(hr_scores)

    return {
        f'ndcg@{k}': avg_ndcg,
        f'hr@{k}': avg_hr
    }
