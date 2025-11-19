"""
Dedicated converter for SASRec models trained with LSQ QAT.

The script loads a QuantizedSASRec checkpoint, applies PyTorch dynamic
quantization to turn learned step-size weights into real INT8 tensors,
and saves the compact state dict for CPU inference.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

try:
    from torch.quantization import quantize_dynamic
except ImportError:  # torch>=2.1 renamed the namespace
    from torch.ao.quantization import quantize_dynamic

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.sasrec_fixed import QuantizedSASRec  # noqa: E402


def _resolve_num_items(checkpoint_dict, config_dict, override):
    """Find the number of catalog items, falling back to CLI override."""
    if override is not None:
        return override

    for key in ('num_items', 'n_items', 'total_items'):
        if key in checkpoint_dict:
            return checkpoint_dict[key]
        if key in config_dict:
            return config_dict[key]

    raise ValueError(
        "Number of items is missing in checkpoint metadata. "
        "Pass --num-items explicitly."
    )


def _instantiate_model(config, num_items, quant_method, bit_width):
    """Create QuantizedSASRec with fallbacks for legacy config fields."""
    max_seq_len = config.get('max_len', config.get('max_seq_len', 200))
    hidden_units = config.get('hidden_units', 128)
    num_blocks = config.get('num_blocks', 2)
    num_heads = config.get('num_heads', 2)
    dropout = config.get('dropout', config.get('dropout_rate', 0.2))
    ffn_type = config.get('ffn_type', 'swiglu')
    ffn_factor = config.get('ffn_factor', 4)
    use_rel_pos_bias = not config.get('disable_rel_pos_bias', False)

    model = QuantizedSASRec(
        num_items=num_items,
        max_seq_len=max_seq_len,
        hidden_units=hidden_units,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout,
        quantizer_type=quant_method,
        bit_width=bit_width,
        ffn_type=ffn_type,
        ffn_factor=ffn_factor,
        use_rel_pos_bias=use_rel_pos_bias
    )
    return model


def _state_dict_size_mb(state_dict):
    """Persist a state dict temporarily to measure serialized size in MB."""
    fd, tmp_path = tempfile.mkstemp(suffix='.pt')
    os.close(fd)
    try:
        torch.save(state_dict, tmp_path)
        return os.path.getsize(tmp_path) / (1024 * 1024)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def convert_sasrec_lsq_to_int8(
    checkpoint_path: str,
    output_path: Union[str, Path],
    num_items: Optional[int] = None,
    quant_method: str = 'lsq',
    bit_width: int = 8,
    device: Optional[torch.device] = None
):
    """Load the checkpoint, quantize Linear layers, and save INT8 weights."""
    device = device or torch.device('cpu')
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    print(f"[1/4] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get('args') or checkpoint.get('config') or {}
    quant_from_ckpt = (config.get('quantization') or quant_method or 'lsq').lower()

    if quant_method is None:
        quant_method = quant_from_ckpt
    quant_method = quant_method.lower()
    if quant_method.lower() != 'lsq':
        print(f"[WARN] Expected LSQ quantization, got '{quant_method}'. Continuing anyway.")

    resolved_num_items = _resolve_num_items(checkpoint, config, num_items)
    bit_width = config.get('bit_width', bit_width)

    model = _instantiate_model(config, resolved_num_items, quant_method, bit_width)

    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys during load: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys during load: {unexpected}")

    model.eval()

    print("[2/4] Measuring baseline (fake-quant) model size...")
    fp32_size = _state_dict_size_mb(model.state_dict())
    print(f"       -> {fp32_size:.2f} MB")

    print("[3/4] Applying dynamic quantization to Linear layers...")
    model_int8 = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model_int8.eval()

    print(f"[4/4] Saving INT8 state dict to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_int8.state_dict(), output_path)
    int8_size = os.path.getsize(output_path) / (1024 * 1024)

    print("\nConversion summary")
    print("------------------")
    print(f"Checkpoint:     {checkpoint_path.name}")
    print(f"Quant method:   {quant_method.upper()}")
    print(f"Bit width:      {bit_width}")
    print(f"Items:          {resolved_num_items}")
    print(f"FP32 size:      {fp32_size:.2f} MB")
    print(f"INT8 size:      {int8_size:.2f} MB")
    reduction = (fp32_size - int8_size) / fp32_size * 100 if fp32_size else 0.0
    print(f"Size reduction: {reduction:.1f}%")
    print(f"Saved model:    {output_path}")

    return {
        'checkpoint': str(checkpoint_path),
        'output': str(output_path),
        'quantization': quant_method,
        'bit_width': bit_width,
        'num_items': resolved_num_items,
        'fp32_size_mb': fp32_size,
        'int8_size_mb': int8_size,
        'size_reduction_pct': reduction
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a SASRec LSQ checkpoint into a real INT8 model"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the LSQ-trained SASRec checkpoint (.pt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Destination path for the INT8 weights (defaults to results/int8/<name>_int8.pt)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/int8',
        help='Directory for the INT8 artifact when --output is not specified'
    )
    parser.add_argument(
        '--num-items',
        type=int,
        default=None,
        help='Override number of catalog items if not stored in checkpoint'
    )
    parser.add_argument(
        '--bit-width',
        type=int,
        default=8,
        help='Bit width that was used during LSQ training (default: 8)'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        default='lsq',
        help='Stored quantization method (only LSQ is expected)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    cli_args = parse_args()
    checkpoint_name = Path(cli_args.checkpoint).stem
    if cli_args.output:
        destination = Path(cli_args.output)
    else:
        destination = Path(cli_args.output_dir) / f"{checkpoint_name}_int8.pt"

    convert_sasrec_lsq_to_int8(
        checkpoint_path=cli_args.checkpoint,
        output_path=destination,
        num_items=cli_args.num_items,
        quant_method=cli_args.quantization,
        bit_width=cli_args.bit_width
    )
