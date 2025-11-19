"""
Train models using NAS-optimized hyperparameters
Reads optimal parameters from JSON and calls production training scripts
"""
import json
import argparse
import subprocess
import sys
import os


def load_nas_params(nas_results_path):
    """Load NAS optimization results"""
    with open(nas_results_path, 'r') as f:
        results = json.load(f)
    return results['best_params']


def build_training_command(model, method, params, epochs, save_dir, log_dir, seed):
    """Build training command with NAS-optimized parameters"""

    # Map model to training script
    script_map = {
        'lstm': 'scripts/train_lstm_production.py',
        'espcn': 'scripts/train_espcn_production.py',
        'sasrec': 'scripts/train_sasrec_production.py'
    }

    script = script_map.get(model)
    if not script:
        raise ValueError(f"Unknown model: {model}")

    # Build command
    cmd = [
        sys.executable,
        script,
        f"--quantization={method}",
        f"--epochs={epochs}",
        f"--save_dir={save_dir}",
        f"--log_dir={log_dir}",
        f"--seed={seed}"
    ]

    # Add NAS-optimized parameters
    if 'lr' in params:
        cmd.append(f"--lr={params['lr']}")

    if 'batch_size' in params:
        cmd.append(f"--batch_size={params['batch_size']}")

    if 'weight_decay' in params:
        cmd.append(f"--weight_decay={params['weight_decay']}")

    if 'bit_width' in params:
        cmd.append(f"--bit_width={params['bit_width']}")

    # Method-specific parameters
    if method == 'lsq' and 'step_size' in params:
        cmd.append(f"--step_size={params['step_size']}")

    if method == 'pact' and 'alpha' in params:
        cmd.append(f"--alpha={params['alpha']}")

    if method == 'adaround' and 'beta' in params:
        cmd.append(f"--beta={params['beta']}")

    if method == 'apot' and 'levels' in params:
        cmd.append(f"--levels={params['levels']}")

    if method == 'dsq' and 'temperature' in params:
        cmd.append(f"--temperature={params['temperature']}")

    return cmd


def main():
    parser = argparse.ArgumentParser(description='Train with NAS-optimized parameters')
    parser.add_argument('--model', type=str, required=True,
                       choices=['lstm', 'espcn', 'sasrec'],
                       help='Model architecture')
    parser.add_argument('--method', type=str, required=True,
                       choices=['lsq', 'pact', 'adaround', 'apot', 'dsq'],
                       help='Quantization method')
    parser.add_argument('--nas_results', type=str, required=True,
                       help='Path to NAS results JSON file')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Directory to save logs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Load NAS parameters
    print(f"Loading NAS results from: {args.nas_results}")

    if not os.path.exists(args.nas_results):
        print(f"[ERROR] NAS results file not found: {args.nas_results}")
        print("[INFO] Falling back to default parameters")
        # Fall back to production script with defaults
        script_map = {
            'lstm': 'scripts/train_lstm_production.py',
            'espcn': 'scripts/train_espcn_production.py',
            'sasrec': 'scripts/train_sasrec_production.py'
        }
        cmd = [
            sys.executable,
            script_map[args.model],
            f"--quantization={args.method}",
            f"--epochs={args.epochs}",
            "--batch_size=128",
            "--lr=1e-3",
            "--bit_width=8",
            f"--save_dir={args.save_dir}",
            f"--log_dir={args.log_dir}",
            f"--seed={args.seed}"
        ]
    else:
        params = load_nas_params(args.nas_results)
        print(f"[INFO] Loaded NAS-optimized parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        # Build training command
        cmd = build_training_command(
            args.model,
            args.method,
            params,
            args.epochs,
            args.save_dir,
            args.log_dir,
            args.seed
        )

    # Execute training
    print(f"\n[INFO] Executing: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
