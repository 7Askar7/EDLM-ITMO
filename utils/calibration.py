"""Calibration utilities for quantization-aware training.

This module provides calibration functions for initializing quantization
parameters before training. Calibration is critical for advanced quantization
methods that require data-driven initialization.

Supported Quantization Methods:
    - LSQ (Learned Step Size Quantization)
    - APoT (Additive Powers-of-Two Quantization)
    - DSQ (Differentiable Soft Quantization)
    - PACT (Parameterized Clipping Activation)

Calibration Process:
    1. Run forward passes on representative data
    2. Collect activation and weight statistics
    3. Initialize quantization parameters (scales, codebooks, etc.)
    4. Verify initialization completeness

References:
    - "Learned Step Size Quantization" (Esser et al., ICLR 2020)
    - "Additive Powers-of-Two Quantization" (Li et al., CVPR 2020)
    - "Differentiable Soft Quantization" (Gong et al., ICML 2019)
"""
import torch
import torch.nn as nn
from tqdm import tqdm


def calibrate_model(model, dataloader, device='cuda', num_batches=100):
    """Calibrate quantization parameters on representative data.

    Calibration is a critical step for advanced quantization methods that
    require data-driven initialization. It performs forward passes on real
    data to collect statistics and initialize quantization parameters.

    This is essential for:
        1. APoT: Initialize alpha/scale parameters from weight distributions
        2. DSQ: Initialize soft quantization codebooks
        3. PACT: Set proper activation clipping ranges
        4. LSQ: Initialize step sizes based on data range

    Args:
        model: Model with quantization layers to calibrate.
        dataloader: DataLoader providing calibration data (typically a
            subset of training data).
        device: Device to run calibration on (default: 'cuda').
        num_batches: Number of batches to use for calibration (default: 100).
            More batches = better statistics but slower calibration.

    Note:
        The model is set to training mode during calibration to allow
        quantizers to collect statistics. No gradient computation is
        performed.
    """
    print("\n" + "="*80)
    print("CALIBRATION PHASE - Initializing Quantization Parameters")
    print("="*80)

    # Set model to training mode (required for quantizer initialization)
    model.train()

    # Collect statistics without computing gradients
    with torch.no_grad():
        pbar = tqdm(
            dataloader,
            total=min(num_batches, len(dataloader)),
            desc="Calibrating"
        )

        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= num_batches:
                break

            # Unpack batch (assumes SASRec-style batch format)
            log_seqs, pos_seqs, neg_seqs = batch
            log_seqs = log_seqs.to(device)
            pos_seqs = pos_seqs.to(device)
            neg_seqs = neg_seqs.to(device)

            # Forward pass to initialize quantizers
            _ = model(log_seqs, pos_seqs, neg_seqs)

            pbar.set_postfix({'batch': f"{batch_idx+1}/{num_batches}"})

    print("Calibration completed")
    print("="*80 + "\n")

    # Verify that all quantizers are properly initialized
    verify_quantizer_initialization(model)


def verify_quantizer_initialization(model):
    """Verify that all quantizers in the model are properly initialized.

    Checks all quantization layers in the model to ensure they have been
    properly initialized during calibration. Prints a summary of initialized
    and uninitialized quantizers.

    Args:
        model: Model to verify.

    Returns:
        bool: True if all quantizers are initialized, False otherwise.

    Note:
        Only APoT and DSQ quantizers have explicit initialization flags.
        LSQ and PACT are assumed to be initialized if present.
    """
    from quantization.apot import APoTQuantize
    from quantization.dsq import DSQQuantize
    from quantization.lsq import LSQQuantizer
    from quantization.pact import PACTQuantizer

    uninitialized = []
    initialized = []

    for name, module in model.named_modules():
        if isinstance(module, (APoTQuantize, DSQQuantize)):
            if hasattr(module, 'initialized'):
                if module.initialized.item() == 0:
                    uninitialized.append(f"{name} ({type(module).__name__})")
                else:
                    initialized.append(f"{name} ({type(module).__name__})")
        elif isinstance(module, (LSQQuantizer, PACTQuantizer)):
            # LSQ and PACT don't have explicit initialization flag
            initialized.append(f"{name} ({type(module).__name__})")

    if initialized:
        print(f"✓ Initialized quantizers: {len(initialized)}")

    if uninitialized:
        print(f"⚠ WARNING: Uninitialized quantizers found: {len(uninitialized)}")
        for q in uninitialized[:5]:  # Show first 5
            print(f"  - {q}")
        if len(uninitialized) > 5:
            print(f"  ... and {len(uninitialized) - 5} more")
        return False

    return True


def get_quantizer_statistics(model):
    """
    Get statistics about quantization parameters for logging
    """
    from quantization.apot import APoTQuantize
    from quantization.dsq import DSQQuantize
    from quantization.lsq import LSQQuantizer

    stats = {
        'apot_alphas': [],
        'dsq_scales': [],
        'dsq_temperatures': [],
        'lsq_steps': []
    }

    for name, module in model.named_modules():
        if isinstance(module, APoTQuantize):
            if hasattr(module, 'alpha'):
                stats['apot_alphas'].append(module.alpha.item())
        elif isinstance(module, DSQQuantize):
            if hasattr(module, 'scale'):
                stats['dsq_scales'].append(module.scale.item())
            if hasattr(module, 'temperature'):
                stats['dsq_temperatures'].append(module.temperature.item())
        elif isinstance(module, LSQQuantizer):
            if hasattr(module, 's'):
                stats['lsq_steps'].append(module.s.item())

    # Compute summaries
    summary = {}
    for key, values in stats.items():
        if values:
            summary[f'{key}_mean'] = sum(values) / len(values)
            summary[f'{key}_min'] = min(values)
            summary[f'{key}_max'] = max(values)

    return summary
