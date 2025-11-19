"""Production training script for ESPCN super-resolution with BSD dataset.

This module implements a production-ready training pipeline for ESPCN-based
image super-resolution with support for multiple quantization methods.

Features:
    - Real BSD300/DIV2K dataset loading and processing
    - Multiple quantization methods (LSQ, PACT, AdaRound, APoT, DSQ)
    - Comprehensive logging and monitoring
    - GPU acceleration support
    - Automatic mixed precision training
    - Best model checkpointing with PSNR metric
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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.espcn import ESPCN, QuantizedESPCN, calculate_psnr
from utils.data_loader import load_bsd_dataset, get_dataloader
from utils.data_loader_div2k import get_div2k_dataloaders
from utils.metrics import calculate_psnr_batch


def setup_logging(log_dir):
    """Configure logging with file and console handlers.

    Args:
        log_dir: Directory path for storing log files.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_epoch(model, dataloader, criterion, optimizer, device,
                use_amp=False, logger=None, regularization_weight=0.0):
    """Train model for one epoch.

    Args:
        model: PyTorch model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Device to run training on (CPU/GPU).
        use_amp: Whether to use automatic mixed precision.
        logger: Logger instance for logging metrics.
        regularization_weight: Weight for regularization loss (AdaRound).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        lr_images, hr_images = batch
        lr_images, hr_images = lr_images.to(device), hr_images.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(lr_images)
                loss = criterion(outputs, hr_images)
                if regularization_weight > 0 and hasattr(model, 'regularization_loss'):
                    loss = loss + model.regularization_loss(regularization_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(lr_images)
            loss = criterion(outputs, hr_images)
            if regularization_weight > 0 and hasattr(model, 'regularization_loss'):
                loss = loss + model.regularization_loss(regularization_weight)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})

    avg_loss = total_loss / len(dataloader)

    if logger:
        logger.info(f"Train Loss: {avg_loss:.6f}")

    return avg_loss


def evaluate(model, dataloader, criterion, device, logger=None):
    """Evaluate model on validation/test set.

    Args:
        model: PyTorch model to evaluate.
        dataloader: Validation/test data loader.
        criterion: Loss function.
        device: Device to run evaluation on (CPU/GPU).
        logger: Logger instance for logging metrics.

    Returns:
        tuple: (average_loss, average_psnr)
    """
    model.eval()
    total_loss = 0
    psnr_values = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            lr_images, hr_images = batch
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            outputs = model(lr_images)
            loss = criterion(outputs, hr_images)

            total_loss += loss.item()

            # Calculate PSNR
            batch_psnr = calculate_psnr_batch(outputs, hr_images)
            psnr_values.append(batch_psnr)

    avg_loss = total_loss / len(dataloader)
    avg_psnr = sum(psnr_values) / len(psnr_values)

    if logger:
        logger.info(f"Val Loss: {avg_loss:.6f}, PSNR: {avg_psnr:.2f} dB")

    return avg_loss, avg_psnr


def print_gpu_info(logger):
    """Display detailed GPU information and memory statistics.

    Args:
        logger: Logger instance for output.

    Returns:
        None
    """
    if torch.cuda.is_available():
        logger.info("="*80)
        logger.info("🚀 GPU ACCELERATED TRAINING ENABLED 🚀")
        logger.info("="*80)
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")

        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        reserved_mem = torch.cuda.memory_reserved(0) / 1024**3
        allocated_mem = torch.cuda.memory_allocated(0) / 1024**3

        logger.info(f"Total Memory: {total_mem:.2f} GB")
        logger.info(f"Reserved Memory: {reserved_mem:.2f} GB")
        logger.info(f"Allocated Memory: {allocated_mem:.2f} GB")
        logger.info(f"Free Memory: {total_mem - reserved_mem:.2f} GB")
        logger.info("="*80)
        logger.info("✅ Training will run on GPU - Expect 5-7x speedup!")
        logger.info("="*80)
    else:
        logger.warning("="*80)
        logger.warning("⚠️  WARNING: CUDA NOT AVAILABLE - TRAINING ON CPU!")
        logger.warning("="*80)


def main(args):
    """Main training function.

    Args:
        args: Parsed command-line arguments containing all configuration.

    Returns:
        float: Best PSNR score achieved during training.
    """
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("="*80)
    logger.info("ESPCN Super Resolution with Quantization - Production Training")
    logger.info("="*80)
    logger.info(f"Arguments: {vars(args)}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Print detailed GPU info
    print_gpu_info(logger)

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset - DIV2K if path contains 'DIV2K', else BSD300
    if 'DIV2K' in args.data_dir or 'div2k' in args.data_dir:
        logger.info("Loading DIV2K dataset...")
        train_loader, test_loader = get_div2k_dataloaders(
            data_dir=args.data_dir,
            scale_factor=args.upscale_factor,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers
        )
        logger.info(f"DIV2K dataset loaded successfully")
    else:
        logger.info("Loading BSD300 dataset...")
        train_dataset, test_dataset = load_bsd_dataset(
            data_dir=args.data_dir,
            upscale_factor=args.upscale_factor,
            patch_size=args.patch_size
        )

        logger.info(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

        # Create dataloaders
        train_loader = get_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers
        )

        test_loader = get_dataloader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers
        )

    # Create model
    logger.info(f"Creating model with quantization: {args.quantization}")

    if args.quantization == 'none':
        model = ESPCN(
            upscale_factor=args.upscale_factor,
            num_channels=3,
            feature_channels=args.feature_channels
        )
    else:
        model = QuantizedESPCN(
            upscale_factor=args.upscale_factor,
            num_channels=3,
            feature_channels=args.feature_channels,
            quantizer_type=args.quantization,
            bit_width=args.bit_width
        )

    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    regularization_weight = args.adaround_regularization if args.quantization == 'adaround' else 0.0

    weight_quant_state = True
    activation_quant_state = True
    if isinstance(model, QuantizedESPCN):
        weight_quant_state = args.weight_warmup_epochs <= 0
        activation_quant_state = args.activation_warmup_epochs <= 0
        model.set_weight_quantization(weight_quant_state)
        model.set_activation_quantization(activation_quant_state)

    # Training loop
    best_psnr = 0.0
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_psnr': [],
        'lr': []
    }

    logger.info(f"\nStarting training for {args.epochs} epochs...")
    logger.info("="*80)

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-"*80)

        if isinstance(model, QuantizedESPCN):
            target_weight_state = (epoch + 1) > args.weight_warmup_epochs
            target_activation_state = (epoch + 1) > args.activation_warmup_epochs

            if target_weight_state != weight_quant_state:
                weight_quant_state = target_weight_state
                model.set_weight_quantization(weight_quant_state)
                state_msg = "ENABLED" if weight_quant_state else "DISABLED"
                logger.info(f"Weight quantization {state_msg} at epoch {epoch + 1}")

            if target_activation_state != activation_quant_state:
                activation_quant_state = target_activation_state
                model.set_activation_quantization(activation_quant_state)
                state_msg = "ENABLED" if activation_quant_state else "DISABLED"
                logger.info(f"Activation quantization {state_msg} at epoch {epoch + 1}")

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, args.use_amp, logger,
            regularization_weight=regularization_weight
        )

        # Evaluate
        test_loss, test_psnr = evaluate(model, test_loader, criterion, device, logger)

        # Update scheduler
        scheduler.step(test_psnr)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_psnr'].append(test_psnr)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        logger.info(f"Epoch time: {epoch_time:.2f}s, LR: {current_lr:.6f}")

        # Save best model
        if test_psnr > best_psnr:
            best_psnr = test_psnr
            save_path = os.path.join(
                args.save_dir,
                f'espcn_{args.quantization}_best.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'psnr': best_psnr,
                'args': vars(args)
            }, save_path)
            logger.info(f"[SAVED] Best model with PSNR: {best_psnr:.2f} dB")

        # Early stopping
        if current_lr < 1e-6:
            logger.info("Learning rate too small, stopping training")
            break

    total_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info(f"Training completed in {total_time/60:.2f} minutes")
    logger.info(f"Best Test PSNR: {best_psnr:.2f} dB")
    logger.info("="*80)

    # Save history
    history_path = os.path.join(args.save_dir, f'espcn_{args.quantization}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"History saved to {history_path}")

    # Save final model
    final_save_path = os.path.join(args.save_dir, f'espcn_{args.quantization}_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'best_psnr': best_psnr
    }, final_save_path)

    logger.info(f"Final model saved to {final_save_path}")

    return best_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ESPCN with real dataset (DIV2K or BSD)')

    # Data arguments
    parser.add_argument('--data_dir', type=str,
                        default='./data/DIV2K',
                        help='Directory for DIV2K or BSD data')
    parser.add_argument('--upscale_factor', type=int, default=3,
                        help='Upscaling factor')
    parser.add_argument('--patch_size', type=int, default=96,
                        help='Patch size for training')

    # Model arguments
    parser.add_argument('--feature_channels', type=int, default=64,
                        help='Number of feature channels')

    # Quantization arguments
    parser.add_argument('--quantization', type=str, default='none',
                        choices=['none', 'lsq', 'pact', 'adaround', 'apot', 'dsq'],
                        help='Quantization method')
    parser.add_argument('--bit_width', type=int, default=8,
                        help='Bit width for quantization')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--weight_warmup_epochs', type=int, default=0,
                        help='Disable weight quantization for the first N epochs')
    parser.add_argument('--activation_warmup_epochs', type=int, default=0,
                        help='Disable activation quantization for the first N epochs')
    parser.add_argument('--adaround_regularization', type=float, default=1e-4,
                        help='Regularization strength for AdaRound rounding parameters')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (default 0 for Windows compatibility)')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches to prefetch per worker (only used when num_workers > 0)')
    parser.add_argument('--persistent_workers', action='store_true',
                        help='Keep DataLoader workers alive between epochs (requires num_workers > 0)')
    parser.add_argument('--save_dir', type=str, default='./results/espcn',
                        help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default='./logs/espcn',
                        help='Directory for logs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    main(args)
