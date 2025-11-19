"""Utils package for machine learning utilities.

This package provides utilities for dataset loading, metrics calculation,
model evaluation, and quantization calibration.

Modules:
    datasets: Dataset classes for LSTM, ESPCN, and SASRec models.
    data_loader: Production-ready data loaders with auto-download.
    data_loader_div2k: DIV2K dataset loader for super-resolution.
    metrics: Evaluation metrics (ROC-AUC, PSNR, NDCG, Hit Rate).
    evaluation: Advanced evaluation protocols for recommendation systems.
    calibration: Quantization calibration utilities.
    log_viewer: Interactive log viewer and results analyzer.
"""
from .metrics import (
    calculate_roc_auc,
    calculate_psnr,
    calculate_ndcg_at_k,
    calculate_hit_rate,
    MetricsTracker,
    evaluate_lstm_classifier,
    evaluate_espcn,
    evaluate_sasrec
)

__all__ = [
    'calculate_roc_auc',
    'calculate_psnr',
    'calculate_ndcg_at_k',
    'calculate_hit_rate',
    'MetricsTracker',
    'evaluate_lstm_classifier',
    'evaluate_espcn',
    'evaluate_sasrec'
]
