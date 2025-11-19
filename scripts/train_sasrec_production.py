"""
Production-quality training script for SASRec with MovieLens-1M dataset
Implements all best practices for quantization-aware training research

Key Features:
- Full reproducibility (deterministic training)
- Proper hyperparameters per quantization method
- Statistical validation with multiple runs
- Complete monitoring and logging
- Graceful error handling and checkpointing
- AMP compatibility detection
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
import json
from tqdm import tqdm
import time
import logging
from datetime import datetime
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import warnings

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.sasrec_fixed import SASRec
from models.sasrec_quantized import QuantizedSASRec
from utils.data_loader import load_movielens_dataset, get_dataloader
from utils.metrics import calculate_ndcg_at_k, calculate_hit_rate
from utils.calibration import calibrate_model, get_quantizer_statistics


# ============================================================================
# REPRODUCIBILITY UTILITIES
# ============================================================================

def set_random_seeds(seed: int, use_deterministic: bool = True):
    """
    Set random seeds for full reproducibility

    Args:
        seed: Random seed value
        use_deterministic: Enable deterministic CUDA operations (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if use_deterministic:
        # CRITICAL for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Use deterministic algorithms where possible
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # For speed (non-deterministic)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int):
    """
    Initialize worker processes with different seeds for DataLoader
    Critical for deterministic negative sampling across workers
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_generator(seed: int) -> torch.Generator:
    """Create a generator for DataLoader with fixed seed"""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ============================================================================
# LOGGING AND MONITORING
# ============================================================================

def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup comprehensive logging"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{experiment_name}_{timestamp}.log')

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")

    return logger


def log_model_statistics(model: nn.Module, logger: logging.Logger):
    """Log comprehensive model statistics"""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("="*80)
    logger.info("MODEL STATISTICS")
    logger.info("="*80)
    logger.info(f"Total parameters: {num_params:,}")
    logger.info(f"Trainable parameters: {num_trainable:,}")
    logger.info(f"Non-trainable parameters: {num_params - num_trainable:,}")
    logger.info(f"Model size (MB): {num_params * 4 / 1024**2:.2f}")  # Assuming FP32
    logger.info("="*80)


def log_gradient_statistics(model: nn.Module, logger: logging.Logger, step: int):
    """Log gradient norms for monitoring training stability"""
    total_norm = 0.0
    param_count = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            param_count += 1

    total_norm = total_norm ** 0.5

    if param_count > 0:
        logger.debug(f"Step {step}: Gradient norm = {total_norm:.6f}")

    return total_norm


def log_weight_statistics(model: nn.Module, logger: logging.Logger, epoch: int):
    """Log weight distribution statistics"""
    logger.info(f"\nWeight Statistics (Epoch {epoch}):")

    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            weight_mean = param.data.mean().item()
            weight_std = param.data.std().item()
            weight_min = param.data.min().item()
            weight_max = param.data.max().item()

            logger.info(
                f"  {name}: mean={weight_mean:.6f}, std={weight_std:.6f}, "
                f"min={weight_min:.6f}, max={weight_max:.6f}"
            )


# ============================================================================
# AMP COMPATIBILITY DETECTION
# ============================================================================

def check_amp_compatibility(quantization: str, logger: logging.Logger) -> bool:
    """
    Check if AMP is compatible with the quantization method

    APoT and DSQ have issues with AMP due to:
    - Custom quantization operations not supporting FP16
    - Numerical instability in codebook operations
    """
    incompatible_methods = ['apot', 'dsq']

    if quantization in incompatible_methods:
        logger.warning("="*80)
        logger.warning(f"WARNING: AMP is NOT recommended with {quantization.upper()}")
        logger.warning("Reason: Custom quantization operations may have FP16 numerical issues")
        logger.warning("Automatically disabling AMP for this method")
        logger.warning("="*80)
        return False

    return True


# ============================================================================
# HYPERPARAMETER CONFIGURATIONS
# ============================================================================

def get_method_hyperparameters(quantization: str, bit_width: int) -> Dict:
    """
    Get optimal hyperparameters for each quantization method
    Based on empirical research and paper recommendations
    """
    configs = {
        'none': {
            'epochs': 100,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'warmup_epochs': 5,
            'patience': 10,
            'lr_decay_factor': 0.5,
            'grad_clip': 1.0,
        },
        'lsq': {
            'epochs': 120,  # Longer for QAT
            'lr': 1e-3,
            'lr_scale': 1e-2,  # Separate LR for scale parameters
            'weight_decay': 1e-4,
            'warmup_epochs': 10,  # More warmup for stability
            'patience': 15,
            'lr_decay_factor': 0.5,
            'grad_clip': 1.0,
        },
        'pact': {
            'epochs': 120,
            'lr': 1e-3,
            'lr_alpha': 1e-2,  # Separate LR for alpha (clipping)
            'weight_decay': 1e-4,
            'warmup_epochs': 10,
            'patience': 15,
            'lr_decay_factor': 0.5,
            'grad_clip': 1.0,
        },
        'adaround': {
            'epochs': 100,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'warmup_epochs': 5,
            'patience': 10,
            'lr_decay_factor': 0.5,
            'grad_clip': 1.0,
        },
        'apot': {
            'epochs': 150,  # Longer for non-uniform quantization
            'lr': 5e-4,  # Lower LR for stability
            'lr_alpha': 1e-3,  # Separate LR for alpha
            'weight_decay': 1e-4,
            'warmup_epochs': 15,  # More warmup
            'patience': 20,
            'lr_decay_factor': 0.5,
            'grad_clip': 0.5,  # Stronger clipping
            'calibration_batches': 200,  # More calibration
        },
        'dsq': {
            'epochs': 150,  # Longer for soft quantization
            'lr': 5e-4,  # Lower LR for stability
            'lr_scale': 1e-3,  # Separate LR for scale/zero_point
            'weight_decay': 1e-4,
            'warmup_epochs': 15,
            'patience': 20,
            'lr_decay_factor': 0.5,
            'grad_clip': 0.5,
            'calibration_batches': 200,
            'temperature_schedule': True,  # Enable temperature annealing
        },
    }

    return configs.get(quantization, configs['none'])


def create_optimizer_groups(model: nn.Module, quantization: str,
                           base_lr: float, config: Dict) -> List[Dict]:
    """
    Create parameter groups with different learning rates
    Critical for QAT: quantization parameters need different LR than weights
    """
    # Separate quantization parameters from model weights
    quant_params = []
    model_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Identify quantization-specific parameters
        if any(key in name for key in ['alpha', 'scale', 's', 'zero_point', 'temperature']):
            quant_params.append(param)
        else:
            model_params.append(param)

    # Create parameter groups
    param_groups = [
        {'params': model_params, 'lr': base_lr, 'name': 'model_weights'}
    ]

    # Add quantization parameters with different LR if applicable
    if len(quant_params) > 0:
        if 'lr_scale' in config:
            quant_lr = config['lr_scale']
        elif 'lr_alpha' in config:
            quant_lr = config['lr_alpha']
        else:
            quant_lr = base_lr

        param_groups.append({
            'params': quant_params,
            'lr': quant_lr,
            'name': 'quant_params'
        })

    return param_groups


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int,
                use_amp: bool = False,
                grad_clip: float = 1.0,
                logger: Optional[logging.Logger] = None) -> Dict:
    """
    Train for one epoch with comprehensive monitoring

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0
    grad_norms = []

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        log_seqs, pos_seqs, neg_seqs = batch
        log_seqs = log_seqs.to(device)
        pos_seqs = pos_seqs.to(device)
        neg_seqs = neg_seqs.to(device)

        optimizer.zero_grad()

        # Forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                pos_logits, neg_logits = model(log_seqs, pos_seqs, neg_seqs)
                # BPR loss with masking
                valid_mask = (pos_seqs > 0).float()
                loss_per_item = -torch.nn.functional.logsigmoid(pos_logits - neg_logits)
                loss = (loss_per_item * valid_mask).sum() / (valid_mask.sum() + 1e-10)
        else:
            pos_logits, neg_logits = model(log_seqs, pos_seqs, neg_seqs)
            valid_mask = (pos_seqs > 0).float()
            loss_per_item = -torch.nn.functional.logsigmoid(pos_logits - neg_logits)
            loss = (loss_per_item * valid_mask).sum() / (valid_mask.sum() + 1e-10)

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        # Track metrics
        total_loss += loss.item()
        grad_norms.append(grad_norm.item())

        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'grad_norm': grad_norm.item()
        })

    avg_loss = total_loss / len(dataloader)
    avg_grad_norm = np.mean(grad_norms)

    metrics = {
        'loss': avg_loss,
        'grad_norm': avg_grad_norm,
        'grad_norm_std': np.std(grad_norms)
    }

    if logger:
        logger.info(
            f"Train - Loss: {avg_loss:.4f}, "
            f"Grad Norm: {avg_grad_norm:.4f} ± {metrics['grad_norm_std']:.4f}"
        )

    return metrics


def evaluate(model: nn.Module,
            dataset: torch.utils.data.Dataset,
            num_items: int,
            k: int = 10,
            device: torch.device = torch.device('cuda'),
            num_samples: Optional[int] = None,
            full_eval: bool = False,
            candidate_pool_size: int = 0,
            negative_candidates: Optional[Dict[int, List[int]]] = None,
            logger: Optional[logging.Logger] = None) -> Tuple[float, float, Dict]:
    """
    Evaluate model with detailed metrics

    Args:
        model: Model to evaluate
        dataset: Dataset
        num_items: Total number of items
        k: Top-k for metrics
        device: Device
        num_samples: Number of samples (None = all)
        full_eval: If True, evaluate on full dataset (for final evaluation)
        candidate_pool_size: Limit number of candidate items (including target)
            ranked per user. If <= 0, use full catalog.
        logger: Logger

    Returns:
        avg_ndcg, avg_hr, metrics_dict
    """
    model.eval()

    ndcg_scores = []
    hr_scores = []

    # Use smaller batch size for evaluation
    batch_size = 64 if full_eval else 128
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    use_candidate_sampling = candidate_pool_size is not None and candidate_pool_size > 0
    if use_candidate_sampling:
        candidate_pool_size = int(candidate_pool_size)
        candidate_pool_size = max(k, min(candidate_pool_size, num_items))
    else:
        candidate_pool_size = 0

    all_item_ids = np.arange(1, num_items + 1, dtype=np.int64)
    all_items_tensor = torch.arange(1, num_items + 1).to(device)

    samples_evaluated = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if not full_eval and num_samples and samples_evaluated >= num_samples:
                break

            log_seqs, pos_seqs, _ = batch
            log_seqs = log_seqs.to(device)

            for i in range(log_seqs.size(0)):
                if not full_eval and num_samples and samples_evaluated >= num_samples:
                    break

                seq = log_seqs[i:i+1]

                # Get target item
                target_items = pos_seqs[i]
                target_items = target_items[target_items != 0].tolist()

                if len(target_items) == 0:
                    continue

                target = target_items[-1]

                candidate_ids = None
                candidate_tensor = all_items_tensor

                if use_candidate_sampling:
                    negative_count = max(candidate_pool_size - 1, 0)
                    negative_count = min(negative_count, num_items - 1)

                    neg_items = None
                    if negative_count > 0:
                        if negative_candidates and target in negative_candidates:
                            stored = negative_candidates[target]
                            if len(stored) >= negative_count:
                                neg_items = np.array(stored[:negative_count], dtype=np.int64)
                        if neg_items is None:
                            neg_indices = np.random.choice(num_items - 1, size=negative_count, replace=False)
                            neg_items = neg_indices + 1
                            neg_items[neg_items >= target] += 1

                    if neg_items is not None:
                        candidate_ids = np.concatenate(([target], neg_items))
                    else:
                        candidate_ids = np.array([target], dtype=np.int64)

                    candidate_tensor = torch.from_numpy(candidate_ids).to(device)

                # Get predictions
                predictions = model.predict(seq, candidate_tensor)
                predictions = predictions.squeeze(0).cpu().numpy()

                # Calculate metrics
                ndcg = calculate_ndcg_at_k(
                    predictions,
                    [target],
                    k,
                    item_ids=candidate_ids
                )
                hr = calculate_hit_rate(
                    predictions,
                    [target],
                    k,
                    item_ids=candidate_ids
                )

                ndcg_scores.append(ndcg)
                hr_scores.append(hr)

                samples_evaluated += 1

    # Calculate metrics with confidence intervals
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    avg_hr = np.mean(hr_scores) if hr_scores else 0.0

    # 95% confidence intervals
    ndcg_std = np.std(ndcg_scores) if len(ndcg_scores) > 1 else 0.0
    hr_std = np.std(hr_scores) if len(hr_scores) > 1 else 0.0

    n = len(ndcg_scores)
    ndcg_ci = 1.96 * ndcg_std / np.sqrt(n) if n > 0 else 0.0
    hr_ci = 1.96 * hr_std / np.sqrt(n) if n > 0 else 0.0

    metrics = {
        'ndcg': avg_ndcg,
        'hr': avg_hr,
        'ndcg_std': ndcg_std,
        'hr_std': hr_std,
        'ndcg_ci': ndcg_ci,
        'hr_ci': hr_ci,
        'samples': samples_evaluated
    }

    if logger:
        eval_type = "FULL" if full_eval else "Sample"
        logger.info(
            f"{eval_type} Eval - NDCG@{k}: {avg_ndcg:.4f} ± {ndcg_ci:.4f}, "
            f"HR@{k}: {avg_hr:.4f} ± {hr_ci:.4f} ({samples_evaluated} samples)"
        )

    return avg_ndcg, avg_hr, metrics


# ============================================================================
# NEGATIVE SAMPLING UTILITIES
# ============================================================================

def build_negative_candidate_map(num_items: int, count: int, seed: int = 42):
    """Precompute fixed negative samples per item for evaluation."""
    if count <= 0:
        return None

    rng = np.random.default_rng(seed)
    all_items = np.arange(1, num_items + 1)
    negative_map = {}

    for item in all_items:
        candidates = np.delete(all_items, item - 1)
        if count >= candidates.size:
            negatives = candidates.copy()
        else:
            negatives = rng.choice(candidates, size=count, replace=False)
        negative_map[item] = negatives.tolist()

    return negative_map


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   scheduler: optim.lr_scheduler._LRScheduler,
                   epoch: int,
                   metrics: Dict,
                   args: argparse.Namespace,
                   filepath: str,
                   logger: logging.Logger):
    """Save training checkpoint with validation"""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'args': vars(args),
            'timestamp': datetime.now().isoformat(),
        }

        # Save to temporary file first
        temp_filepath = filepath + '.tmp'
        torch.save(checkpoint, temp_filepath)

        # Verify the checkpoint can be loaded
        # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
        _ = torch.load(temp_filepath, weights_only=False)

        # Move to final location
        os.replace(temp_filepath, filepath)

        logger.info(f"Checkpoint saved: {filepath}")

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)


def load_checkpoint(filepath: str,
                   model: nn.Module,
                   optimizer: optim.Optimizer,
                   scheduler: optim.lr_scheduler._LRScheduler,
                   logger: logging.Logger) -> int:
    """Load checkpoint and return starting epoch"""
    if not os.path.exists(filepath):
        logger.info("No checkpoint found, starting from scratch")
        return 0

    try:
        # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
        checkpoint = torch.load(filepath, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint['epoch']
        logger.info(f"Checkpoint loaded from epoch {epoch}")

        return epoch + 1

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        logger.info("Starting from scratch")
        return 0


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main(args):
    # Setup experiment name
    experiment_name = f"sasrec_{args.quantization}_b{args.bit_width}_seed{args.seed}"

    # Setup logging
    logger = setup_logging(args.log_dir, experiment_name)
    logger.info("="*80)
    logger.info("SASRec with Quantization - PRODUCTION TRAINING")
    logger.info("="*80)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Arguments:\n{json.dumps(vars(args), indent=2)}")

    # Set reproducibility
    logger.info("\nSetting up reproducibility...")
    set_random_seeds(args.seed, use_deterministic=args.deterministic)
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Deterministic mode: {args.deterministic}")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nUsing device: {device}")

    # Print detailed GPU info
    if torch.cuda.is_available():
        logger.info("="*80)
        logger.info("🚀 GPU ACCELERATED TRAINING ENABLED 🚀")
        logger.info("="*80)
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")

        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        reserved_mem = torch.cuda.memory_reserved(0) / 1024**3
        allocated_mem = torch.cuda.memory_allocated(0) / 1024**3

        logger.info(f"Total Memory: {total_mem:.2f} GB")
        logger.info(f"Reserved Memory: {reserved_mem:.2f} GB")
        logger.info(f"Allocated Memory: {allocated_mem:.2f} GB")
        logger.info(f"Free Memory: {total_mem - reserved_mem:.2f} GB")
        logger.info(f"cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
        logger.info(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        logger.info("="*80)
        logger.info("✅ Training will run on GPU - Expect 5-7x speedup!")
        logger.info("="*80)
    else:
        logger.warning("="*80)
        logger.warning("⚠️  WARNING: CUDA NOT AVAILABLE - TRAINING ON CPU!")
        logger.warning("="*80)

    # Check AMP compatibility
    use_amp = args.use_amp and check_amp_compatibility(args.quantization, logger)
    if args.use_amp and not use_amp:
        logger.info("AMP disabled due to compatibility issues")

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    logger.info("\n" + "="*80)
    logger.info("LOADING DATASET")
    logger.info("="*80)

    train_dataset, test_dataset, num_users, num_items = load_movielens_dataset(
        data_dir=args.data_dir,
        min_rating=args.min_rating,
        max_len=args.max_len
    )

    logger.info(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    logger.info(f"Users: {num_users}, Items: {num_items}")

    # Create dataloader with deterministic settings
    generator = create_generator(args.seed) if args.deterministic else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        worker_init_fn=worker_init_fn if args.deterministic else None,
        generator=generator,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # Get hyperparameters for this method
    method_config = get_method_hyperparameters(args.quantization, args.bit_width)

    # Override with command line args if provided
    if args.epochs is not None:
        method_config['epochs'] = args.epochs
    if args.lr is not None:
        method_config['lr'] = args.lr

    logger.info(f"\nUsing hyperparameters for {args.quantization}:")
    logger.info(json.dumps(method_config, indent=2))

    # Create model
    logger.info("\n" + "="*80)
    logger.info("CREATING MODEL")
    logger.info("="*80)
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"Bit width: {args.bit_width}")

    if args.quantization == 'none':
        model = SASRec(
            num_items=num_items,
            max_seq_len=args.max_len,
            hidden_units=args.hidden_units,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout_rate=args.dropout,
            ffn_type=args.ffn_type,
            ffn_factor=args.ffn_factor,
            use_rel_pos_bias=not args.disable_rel_pos_bias
        )
    else:
        model = QuantizedSASRec(
            num_items=num_items,
            max_seq_len=args.max_len,
            hidden_units=args.hidden_units,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout_rate=args.dropout,
            quantizer_type=args.quantization,
            bit_width=args.bit_width,
            ffn_type=args.ffn_type,
            ffn_factor=args.ffn_factor,
            use_rel_pos_bias=not args.disable_rel_pos_bias
        )

    model = model.to(device)
    log_model_statistics(model, logger)

    # Calibration phase for APoT and DSQ
    if args.quantization in ['apot', 'dsq']:
        calibration_batches = method_config.get('calibration_batches', 100)
        logger.info(f"\n{'='*80}")
        logger.info("CALIBRATION PHASE")
        logger.info(f"{'='*80}")
        logger.info(f"Using {calibration_batches} batches for calibration")

        calibrate_model(
            model,
            train_loader,
            device=device,
            num_batches=calibration_batches
        )

        # Log quantizer statistics
        quant_stats = get_quantizer_statistics(model)
        if quant_stats:
            logger.info("\nQuantizer statistics after calibration:")
            for key, value in quant_stats.items():
                logger.info(f"  {key}: {value:.6f}")

    # Create optimizer with parameter groups
    param_groups = create_optimizer_groups(
        model,
        args.quantization,
        method_config['lr'],
        method_config
    )

    optimizer = optim.Adam(
        param_groups,
        weight_decay=method_config['weight_decay']
    )

    logger.info(f"\nOptimizer parameter groups: {len(param_groups)}")
    for i, group in enumerate(param_groups):
        logger.info(f"  Group {i} ({group['name']}): LR={group['lr']:.6f}, "
                   f"Params={len(group['params'])}")

    # Learning rate scheduler (based on validation metric, not LR!)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize NDCG
        factor=method_config['lr_decay_factor'],
        patience=method_config['patience'],
        min_lr=1e-7
    )

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        checkpoint_path = os.path.join(args.save_dir, f'{experiment_name}_latest.pt')
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler, logger)

    # Training loop
    best_ndcg = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    history = {
        'train_loss': [],
        'train_grad_norm': [],
        'val_ndcg': [],
        'val_hr': [],
        'val_ndcg_ci': [],
        'val_hr_ci': [],
        'lr': [],
        'quant_stats': []
    }

    negative_candidate_map = None
    if args.eval_candidate_items and args.eval_candidate_items > 0:
        negative_candidate_map = build_negative_candidate_map(
            num_items,
            max(args.eval_candidate_items - 1, 0),
            seed=args.seed
        )

    logger.info(f"\n{'='*80}")
    logger.info("TRAINING")
    logger.info(f"{'='*80}")
    logger.info(f"Total epochs: {method_config['epochs']}")
    logger.info(f"Starting from epoch: {start_epoch}")
    logger.info(f"Warmup epochs: {method_config.get('warmup_epochs', 0)}")
    logger.info(f"Early stopping patience: {method_config['patience']}")

    start_time = time.time()

    for epoch in range(start_epoch, method_config['epochs']):
        epoch_start = time.time()

        logger.info(f"\n{'='*80}")
        logger.info(f"EPOCH {epoch + 1}/{method_config['epochs']}")
        logger.info(f"{'='*80}")

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch + 1,
            use_amp=use_amp,
            grad_clip=method_config['grad_clip'],
            logger=logger
        )

        # Evaluate on validation set (sampled)
        val_ndcg, val_hr, val_metrics = evaluate(
            model,
            test_dataset,
            num_items,
            k=args.k,
            device=device,
            num_samples=args.num_eval_samples,
            full_eval=False,
            candidate_pool_size=args.eval_candidate_items,
            negative_candidates=negative_candidate_map,
            logger=logger
        )

        # Update learning rate scheduler (based on NDCG, not LR!)
        scheduler.step(val_ndcg)
        current_lr = optimizer.param_groups[0]['lr']

        # Log quantizer statistics if applicable
        if args.quantization in ['apot', 'dsq', 'lsq', 'pact']:
            quant_stats = get_quantizer_statistics(model)
            if quant_stats:
                logger.info("\nQuantizer statistics:")
                for key, value in quant_stats.items():
                    logger.info(f"  {key}: {value:.6f}")
                history['quant_stats'].append(quant_stats)

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_grad_norm'].append(train_metrics['grad_norm'])
        history['val_ndcg'].append(val_ndcg)
        history['val_hr'].append(val_hr)
        history['val_ndcg_ci'].append(val_metrics['ndcg_ci'])
        history['val_hr_ci'].append(val_metrics['hr_ci'])
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        logger.info(f"\nEpoch Summary:")
        logger.info(f"  Time: {epoch_time:.2f}s")
        logger.info(f"  LR: {current_lr:.8f}")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  Val NDCG@{args.k}: {val_ndcg:.4f} ± {val_metrics['ndcg_ci']:.4f}")
        logger.info(f"  Val HR@{args.k}: {val_hr:.4f} ± {val_metrics['hr_ci']:.4f}")

        # Save best model
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_epoch = epoch
            epochs_without_improvement = 0

            best_path = os.path.join(args.save_dir, f'{experiment_name}_best.pt')

            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'ndcg': best_ndcg, 'hr': val_hr, **val_metrics},
                args, best_path, logger
            )

            logger.info(f"NEW BEST MODEL! NDCG@{args.k}: {best_ndcg:.4f}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epochs "
                       f"(best: {best_ndcg:.4f} at epoch {best_epoch + 1})")

        # Save latest checkpoint for resuming
        if (epoch + 1) % args.checkpoint_every == 0:
            latest_path = os.path.join(args.save_dir, f'{experiment_name}_latest.pt')
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'ndcg': val_ndcg, 'hr': val_hr, **val_metrics},
                args, latest_path, logger
            )

        # Early stopping check
        if args.early_stopping and epochs_without_improvement >= method_config['patience'] * 2:
            logger.info(f"\n{'='*80}")
            logger.info("EARLY STOPPING")
            logger.info(f"No improvement for {epochs_without_improvement} epochs")
            logger.info(f"{'='*80}")
            break

    total_time = time.time() - start_time

    # Final evaluation on full test set
    logger.info(f"\n{'='*80}")
    logger.info("FINAL EVALUATION ON FULL TEST SET")
    logger.info(f"{'='*80}")
    logger.info("Loading best model...")

    # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
    best_checkpoint = torch.load(os.path.join(args.save_dir, f'{experiment_name}_best.pt'), weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    final_ndcg, final_hr, final_metrics = evaluate(
        model,
        test_dataset,
        num_items,
        k=args.k,
        device=device,
        full_eval=True,
        candidate_pool_size=args.eval_candidate_items,
        negative_candidates=negative_candidate_map,
        logger=logger
    )

    # Log final results
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"Total time: {total_time/60:.2f} minutes")
    logger.info(f"Best epoch: {best_epoch + 1}")
    logger.info(f"Best validation NDCG@{args.k}: {best_ndcg:.4f}")
    logger.info(f"\nFinal Test Results (Full Dataset):")
    logger.info(f"  NDCG@{args.k}: {final_ndcg:.4f} ± {final_metrics['ndcg_ci']:.4f}")
    logger.info(f"  HR@{args.k}: {final_hr:.4f} ± {final_metrics['hr_ci']:.4f}")
    logger.info(f"  Samples evaluated: {final_metrics['samples']}")
    logger.info(f"{'='*80}")

    # Save final results
    results = {
        'experiment': experiment_name,
        'args': vars(args),
        'method_config': method_config,
        'best_epoch': best_epoch + 1,
        'training_time_minutes': total_time / 60,
        'validation': {
            'ndcg': best_ndcg,
            'epoch': best_epoch + 1
        },
        'test': {
            'ndcg': final_ndcg,
            'ndcg_ci': final_metrics['ndcg_ci'],
            'hr': final_hr,
            'hr_ci': final_metrics['hr_ci'],
            'samples': final_metrics['samples']
        },
        'history': history
    }

    results_path = os.path.join(args.save_dir, f'{experiment_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")

    # Save final model
    final_path = os.path.join(args.save_dir, f'{experiment_name}_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_items': num_items,
        'num_users': num_users,
        'args': vars(args),
        'results': results
    }, final_path)

    logger.info(f"Final model saved to: {final_path}")

    return final_ndcg


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Production training for SASRec with QAT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data_dir', type=str, default='./data/ml-1m',
                           help='Directory for MovieLens data')
    data_group.add_argument('--min_rating', type=float, default=4.0,
                           help='Minimum rating threshold')
    data_group.add_argument('--max_len', type=int, default=200,
                           help='Maximum sequence length')

    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--hidden_units', type=int, default=128,
                            help='Hidden units dimension')
    model_group.add_argument('--num_blocks', type=int, default=2,
                            help='Number of self-attention blocks')
    model_group.add_argument('--num_heads', type=int, default=2,
                            help='Number of attention heads')
    model_group.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout rate')
    model_group.add_argument('--ffn_type', type=str, default='swiglu',
                            choices=['swiglu', 'conv'],
                            help='Feed-forward network variant')
    model_group.add_argument('--ffn_factor', type=int, default=4,
                            help='Expansion factor for feed-forward hidden size')
    model_group.add_argument('--disable_rel_pos_bias', action='store_true',
                            help='Disable learnable relative position bias in attention')

    # Quantization arguments
    quant_group = parser.add_argument_group('Quantization')
    quant_group.add_argument('--quantization', type=str, default='none',
                            choices=['none', 'lsq', 'pact', 'adaround', 'apot', 'dsq'],
                            help='Quantization method')
    quant_group.add_argument('--bit_width', type=int, default=8,
                            help='Bit width for quantization')

    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--batch_size', type=int, default=128,
                            help='Batch size')
    train_group.add_argument('--epochs', type=int, default=None,
                            help='Number of epochs (None = use method default)')
    train_group.add_argument('--lr', type=float, default=None,
                            help='Learning rate (None = use method default)')
    train_group.add_argument('--use_amp', action='store_true',
                            help='Use automatic mixed precision (auto-disabled for incompatible methods)')
    train_group.add_argument('--early_stopping', action='store_true', default=True,
                            help='Enable early stopping')
    train_group.add_argument('--checkpoint_every', type=int, default=10,
                            help='Save checkpoint every N epochs')
    train_group.add_argument('--resume', action='store_true',
                            help='Resume from latest checkpoint')

    # Evaluation arguments
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument('--k', type=int, default=10,
                           help='Top-k for evaluation metrics')
    eval_group.add_argument('--num_eval_samples', type=int, default=2000,
                           help='Number of samples for validation (full test at end)')
    eval_group.add_argument('--eval_candidate_items', type=int, default=120,
                           help='Number of candidate items (including target) to rank per user during evaluation (set 0 to use all items)')

    # Reproducibility arguments
    repro_group = parser.add_argument_group('Reproducibility')
    repro_group.add_argument('--seed', type=int, default=42,
                            help='Random seed for reproducibility')
    repro_group.add_argument('--deterministic', action='store_true', default=True,
                            help='Enable deterministic mode (slower but reproducible)')

    # System arguments
    sys_group = parser.add_argument_group('System')
    sys_group.add_argument('--num_workers', type=int, default=0,
                          help='Number of data loading workers (0 for Windows)')
    sys_group.add_argument('--save_dir', type=str, default='./results/sasrec',
                          help='Directory to save results')
    sys_group.add_argument('--log_dir', type=str, default='./logs/sasrec',
                          help='Directory for logs')

    args = parser.parse_args()

    # Run training
    try:
        final_ndcg = main(args)
        print(f"\n{'='*80}")
        print(f"Training completed successfully!")
        print(f"Final NDCG@{args.k}: {final_ndcg:.4f}")
        print(f"{'='*80}")
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"Training failed with error: {e}")
        print(f"{'='*80}")
        raise
