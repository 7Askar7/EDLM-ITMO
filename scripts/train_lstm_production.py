"""Production training script for LSTM text classifier with IMDB dataset.

This module implements a production-ready training pipeline for LSTM-based
text classification with support for multiple quantization methods.

Features:
    - Real IMDB dataset loading and processing
    - Multiple quantization methods (LSQ, PACT, AdaRound, APoT, DSQ)
    - Comprehensive logging and monitoring
    - GPU acceleration support
    - Automatic mixed precision training
    - Best model checkpointing
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

from models.lstm_classifier import LSTMClassifier, QuantizedLSTMClassifier
from utils.data_loader import load_imdb_dataset, get_dataloader
from utils.metrics import calculate_roc_auc


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
                use_amp=False, logger=None):
    """Train model for one epoch.

    Args:
        model: PyTorch model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Device to run training on (CPU/GPU).
        use_amp: Whether to use automatic mixed precision.
        logger: Logger instance for logging metrics.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        texts, lengths, labels = batch
        texts = texts.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(texts, lengths)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})

    avg_loss = total_loss / len(dataloader)

    if logger:
        logger.info(f"Train Loss: {avg_loss:.4f}")

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
        tuple: (average_loss, roc_auc_score)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            texts, lengths, labels = batch
            texts = texts.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Get predictions
            probs = torch.sigmoid(outputs)
            all_predictions.append(probs.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader)

    # Calculate ROC-AUC
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    roc_auc = calculate_roc_auc(all_labels, all_predictions)

    if logger:
        logger.info(f"Val Loss: {avg_loss:.4f}, ROC-AUC: {roc_auc:.4f}")

    return avg_loss, roc_auc


def load_pretrained_weights(model, checkpoint_path, device, logger):
    """Load pretrained FP32 weights for quantized training initialization.

    Args:
        model: Model to load weights into.
        checkpoint_path: Path to FP32 checkpoint file.
        device: Device for loading checkpoint.
        logger: Logger instance.

    Returns:
        None
    """
    if not checkpoint_path:
        return

    if not os.path.exists(checkpoint_path):
        logger.warning(f"Pretrained checkpoint not found at {checkpoint_path}")
        return

    logger.info(f"Loading pretrained FP32 weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys while loading pretrained weights: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys while loading pretrained weights: {unexpected}")

    logger.info("Pretrained weights loaded successfully")


def collect_quantizer_parameters(model):
    """Separate model parameters into base and quantizer-specific groups.

    Args:
        model: Model with quantization layers.

    Returns:
        tuple: (base_params, quant_params) lists of parameters.
    """
    base_params = []
    quant_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'quantizer' in name:
            quant_params.append(param)
        else:
            base_params.append(param)

    return base_params, quant_params


def set_activation_quantization(model, enabled=True):
    """Enable or disable activation quantization in all quantized layers.

    Args:
        model: Model with quantization layers.
        enabled: Whether to enable (True) or disable (False) quantization.

    Returns:
        int: Number of layers affected.
    """
    count = 0
    for module in model.modules():
        if hasattr(module, 'quantize_activation_flag'):
            module.quantize_activation_flag = enabled
            count += 1
    return count


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
        float: Best ROC-AUC score achieved during training.
    """
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("="*80)
    logger.info("LSTM Text Classification with Quantization - Production Training")
    logger.info("="*80)
    logger.info(f"Arguments: {vars(args)}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Print detailed GPU info
    print_gpu_info(logger)

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load IMDB dataset
    logger.info("Loading IMDB dataset...")
    train_dataset, test_dataset, vocab_size = load_imdb_dataset(
        data_dir=args.data_dir,
        max_vocab_size=args.vocab_size,
        max_len=args.max_len
    )

    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    test_loader = get_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create model
    logger.info(f"Creating model with quantization: {args.quantization}")

    if args.quantization == 'none':
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional
        )
    else:
        model = QuantizedLSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            quantizer_type=args.quantization,
            bit_width=args.bit_width
        )

    model = model.to(device)

    if args.quantization != 'none' and args.pretrained_fp32:
        load_pretrained_weights(model, args.pretrained_fp32, device, logger)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()

    if args.quantization != 'none':
        base_params, quant_params = collect_quantizer_parameters(model)
        optimizer_groups = [{
            'params': base_params,
            'lr': args.lr,
            'weight_decay': args.weight_decay
        }]

        if quant_params:
            optimizer_groups.append({
                'params': quant_params,
                'lr': args.quant_param_lr,
                'weight_decay': 0.0
            })

        optimizer = optim.Adam(optimizer_groups)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    activation_quant_enabled = True
    if args.quantization != 'none' and args.quant_activation_warmup > 0:
        disabled = set_activation_quantization(model, enabled=False)
        activation_quant_enabled = False
        logger.info(
            f"Activation quantization disabled for warmup ({args.quant_activation_warmup} epochs). "
            f"Layers affected: {disabled}"
        )

    # Training loop
    best_auc = 0.0
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_auc': [],
        'lr': []
    }

    logger.info(f"\nStarting training for {args.epochs} epochs...")
    logger.info("="*80)

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-"*80)

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, args.use_amp, logger
        )

        # Evaluate
        test_loss, test_auc = evaluate(model, test_loader, criterion, device, logger)

        # Update scheduler
        scheduler.step(test_auc)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_auc'].append(test_auc)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        logger.info(f"Epoch time: {epoch_time:.2f}s, LR: {current_lr:.6f}")

        if (args.quantization != 'none' and not activation_quant_enabled and
                args.quant_activation_warmup > 0 and epoch + 1 >= args.quant_activation_warmup):
            enabled_layers = set_activation_quantization(model, enabled=True)
            activation_quant_enabled = True
            logger.info(f"Activation quantization enabled after warmup. Layers updated: {enabled_layers}")

        # Save best model
        if test_auc > best_auc:
            best_auc = test_auc
            save_path = os.path.join(
                args.save_dir,
                f'lstm_{args.quantization}_best.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'auc': best_auc,
                'vocab_size': vocab_size,
                'args': vars(args)
            }, save_path)
            logger.info(f"[SAVED] Best model with AUC: {best_auc:.4f}")

        # Early stopping
        if current_lr < 1e-6:
            logger.info("Learning rate too small, stopping training")
            break

    total_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info(f"Training completed in {total_time/60:.2f} minutes")
    logger.info(f"Best Test ROC-AUC: {best_auc:.4f}")
    logger.info("="*80)

    # Save history
    history_path = os.path.join(args.save_dir, f'lstm_{args.quantization}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"History saved to {history_path}")

    # Save final model
    final_save_path = os.path.join(args.save_dir, f'lstm_{args.quantization}_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'args': vars(args),
        'best_auc': best_auc
    }, final_save_path)

    logger.info(f"Final model saved to {final_save_path}")

    return best_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM classifier with real IMDB data')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data/imdb',
                        help='Directory for IMDB data')
    parser.add_argument('--vocab_size', type=int, default=20000,
                        help='Vocabulary size')
    parser.add_argument('--max_len', type=int, default=256,
                        help='Maximum sequence length')

    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='Use bidirectional LSTM')

    # Quantization arguments
    parser.add_argument('--quantization', type=str, default='none',
                        choices=['none', 'lsq', 'pact', 'adaround', 'apot', 'dsq'],
                        help='Quantization method')
    parser.add_argument('--bit_width', type=int, default=8,
                        help='Bit width for quantization')
    parser.add_argument('--pretrained_fp32', type=str, default='',
                        help='Path to pretrained FP32 checkpoint for QAT initialization')
    parser.add_argument('--quant_param_lr', type=float, default=1e-3,
                        help='Learning rate for quantizer-specific parameters (e.g., PACT alpha)')
    parser.add_argument('--quant_activation_warmup', type=int, default=0,
                        help='Number of initial epochs to keep activation quantization disabled')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (default 0 for Windows compatibility)')
    parser.add_argument('--save_dir', type=str, default='./results/lstm',
                        help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default='./logs/lstm',
                        help='Directory for logs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    main(args)
