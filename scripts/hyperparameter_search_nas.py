"""
Advanced Hyperparameter Search using Optuna (NAS-like approach)
Efficiently finds optimal hyperparameters for quantization methods
"""
import os
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from tabulate import tabulate
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class QuantizationOptimizer:
    """
    NAS-like optimizer for quantization hyperparameters using Optuna
    """

    def __init__(
        self,
        model_type,
        method,
        epochs=30,
        timeout_hours=2,
        batch_multiplier=1.0,
        max_batch_size=None,
        num_workers=0,
        use_amp=False
    ):
        self.model_type = model_type
        self.method = method
        self.epochs = epochs
        self.timeout_hours = timeout_hours
        self.best_metric = None
        self.trials_data = []
        self.batch_multiplier = batch_multiplier
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers
        self.use_amp = use_amp

        # Setup logging
        self.logger = self._setup_logging()

        # Define search spaces for different methods
        self.search_spaces = self._define_search_spaces()

    def _setup_logging(self):
        """Setup logging for optimization process"""
        log_dir = 'logs/nas_optimization'
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f'{log_dir}/{self.model_type}_{self.method}_{timestamp}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def _define_search_spaces(self):
        """Define hyperparameter search spaces for each method"""

        # Base search space
        base_space = {
            'learning_rate': {
                'type': 'loguniform',
                'low': 1e-5,
                'high': 1e-2
            },
            'batch_size': {
                'type': 'categorical',
                'choices': [16, 32, 64, 128, 256]
            },
            'weight_decay': {
                'type': 'loguniform',
                'low': 1e-6,
                'high': 1e-3
            },
            'dropout': {
                'type': 'uniform',
                'low': 0.1,
                'high': 0.5
            }
        }

        # Method-specific parameters
        method_specific = {
            'lsq': {
                'gradient_scale_factor': {
                    'type': 'loguniform',
                    'low': 0.01,
                    'high': 10.0
                },
                'init_method': {
                    'type': 'categorical',
                    'choices': ['max', 'percentile', 'mse']
                }
            },
            'pact': {
                'alpha_init': {
                    'type': 'uniform',
                    'low': 1.0,
                    'high': 10.0
                },
                'alpha_lr_scale': {
                    'type': 'loguniform',
                    'low': 0.01,
                    'high': 10.0
                }
            },
            'adaround': {
                'calibration_batches': {
                    'type': 'int',
                    'low': 20,
                    'high': 200
                },
                'temperature_init': {
                    'type': 'uniform',
                    'low': 0.5,
                    'high': 2.0
                },
                'temperature_decay': {
                    'type': 'uniform',
                    'low': 0.9,
                    'high': 0.99
                }
            },
            'apot': {
                'calibration_batches': {
                    'type': 'int',
                    'low': 20,
                    'high': 200
                },
                'num_terms': {
                    'type': 'int',
                    'low': 2,
                    'high': 4
                },
                'alpha_init_scale': {
                    'type': 'uniform',
                    'low': 0.5,
                    'high': 2.0
                }
            },
            'dsq': {
                'calibration_batches': {
                    'type': 'int',
                    'low': 20,
                    'high': 200
                },
                'temperature_init': {
                    'type': 'uniform',
                    'low': 0.5,
                    'high': 2.0
                },
                'temperature_min': {
                    'type': 'loguniform',
                    'low': 0.001,
                    'high': 0.1
                }
            }
        }

        # Combine base and method-specific
        spaces = {}
        for method_name, method_params in method_specific.items():
            spaces[method_name] = {**base_space, **method_params}

        # Add base space for methods without specific params
        spaces['none'] = base_space

        return spaces

    def objective(self, trial):
        """
        Objective function for Optuna optimization

        Args:
            trial: Optuna trial object

        Returns:
            float: Metric value (higher is better)
        """

        # Sample hyperparameters from search space
        params = self._sample_hyperparameters(trial)

        # Log trial
        self.logger.info(f"Trial {trial.number}: {params}")

        # Run training with sampled parameters
        result = self._run_training(params)

        # Store trial data
        self.trials_data.append({
            'trial': trial.number,
            'params': params,
            'metric': result['metric'],
            'status': result['status']
        })

        # Report intermediate result for pruning
        if result['status'] == 'success' and 'intermediate_metrics' in result:
            for step, metric_value in enumerate(result['intermediate_metrics']):
                trial.report(metric_value, step)

                # Check if trial should be pruned
                if trial.should_prune():
                    self.logger.info(f"Trial {trial.number} pruned at step {step}")
                    raise optuna.TrialPruned()

        # Return metric (Optuna maximizes by default)
        if result['status'] == 'success':
            return result['metric']
        else:
            # Return worst possible value for failed trials
            return -float('inf') if self.model_type != 'espcn' else 0.0

    def _sample_hyperparameters(self, trial):
        """Sample hyperparameters based on search space definition"""

        params = {}
        search_space = self.search_spaces.get(self.method, self.search_spaces['none'])

        for param_name, param_config in search_space.items():
            if param_config['type'] == 'uniform':
                params[param_name] = trial.suggest_uniform(
                    param_name, param_config['low'], param_config['high']
                )
            elif param_config['type'] == 'loguniform':
                params[param_name] = trial.suggest_loguniform(
                    param_name, param_config['low'], param_config['high']
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high']
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config['choices']
                )

        return params

    def _run_training(self, params):
        """
        Run training with given hyperparameters

        Args:
            params: Dictionary of hyperparameters

        Returns:
            dict: Results including metric and status
        """

        # Map model to script
        script_map = {
            'lstm': 'train_lstm_production.py',
            'espcn': 'train_espcn_production.py',
            'sasrec': 'train_sasrec_production.py'
        }

        script = script_map[self.model_type]
        script_path = os.path.join(os.path.dirname(__file__), script)

        # Build command - use sys.executable to ensure same Python environment
        base_batch_size = int(params['batch_size'])
        scaled_batch_size = max(1, int(base_batch_size * self.batch_multiplier))
        if self.max_batch_size is not None:
            scaled_batch_size = min(scaled_batch_size, int(self.max_batch_size))

        cmd = [
            sys.executable, script_path,
            '--quantization', self.method,
            '--epochs', str(self.epochs),
            '--batch_size', str(scaled_batch_size),
            '--lr', str(params['learning_rate']),
            '--weight_decay', str(params.get('weight_decay', 1e-4)),
            '--seed', '42',
            '--save_dir', f'results/nas_optimization/{self.model_type}',
            '--log_dir', f'logs/nas_optimization/{self.model_type}'
        ]

        cmd.extend(['--num_workers', str(self.num_workers)])
        if self.use_amp:
            cmd.append('--use_amp')

        # Add method-specific parameters
        if 'calibration_batches' in params:
            cmd.extend(['--calibration_batches', str(int(params['calibration_batches']))])

        # Add bit width for quantization methods
        if self.method != 'none':
            cmd.extend(['--bit_width', '8'])

        try:
            # Run training
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_hours*3600)

            # Parse results
            history_file = f'results/nas_optimization/{self.model_type}/{self.model_type}_{self.method}_history.json'

            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)

                # Get metric
                metric_map = {
                    'lstm': ('test_auc', 'max'),
                    'espcn': ('test_psnr', 'max'),
                    'sasrec': ('test_ndcg', 'max')
                }

                metric_key, metric_op = metric_map[self.model_type]

                if metric_key in history:
                    metrics = history[metric_key]
                    if metric_op == 'max':
                        best_metric = max(metrics)
                    else:
                        best_metric = min(metrics)

                    return {
                        'metric': best_metric,
                        'intermediate_metrics': metrics,
                        'status': 'success'
                    }

            return {'metric': 0.0, 'status': 'failed'}

        except subprocess.TimeoutExpired:
            self.logger.warning("Training timeout")
            return {'metric': 0.0, 'status': 'timeout'}
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return {'metric': 0.0, 'status': 'error'}

    def optimize(self, n_trials=50, n_jobs=1):
        """
        Run Optuna optimization

        Args:
            n_trials: Number of trials to run
            n_jobs: Number of parallel jobs

        Returns:
            dict: Best parameters and results
        """

        # Create study with appropriate sampler and pruner
        study_name = f'{self.model_type}_{self.method}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        # Use TPE sampler (Tree-structured Parzen Estimator) - good for hyperparameter optimization
        sampler = TPESampler(seed=42)

        # Use Hyperband pruner for early stopping of bad trials
        pruner = HyperbandPruner(
            min_resource=5,
            max_resource=self.epochs,
            reduction_factor=3
        )

        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            storage=f'sqlite:///results/nas_optimization/optuna_{self.model_type}_{self.method}.db',
            load_if_exists=True
        )

        # Optimize
        self.logger.info(f"Starting optimization with {n_trials} trials")
        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=self.timeout_hours * 3600 * n_trials,
            show_progress_bar=True
        )

        # Get best results
        best_params = study.best_params
        best_value = study.best_value

        self.logger.info(f"Best trial: {study.best_trial.number}")
        self.logger.info(f"Best value: {best_value}")
        self.logger.info(f"Best params: {best_params}")

        # Save results
        results = {
            'model': self.model_type,
            'method': self.method,
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(study.trials),
            'study_name': study_name,
            'trials': self.trials_data
        }

        # Save to JSON
        output_file = f'results/nas_optimization/{self.model_type}_{self.method}_best.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results


def run_nas_optimization(args):
    """
    Run NAS-like optimization for all models and methods
    """

    # Create results directory
    os.makedirs('results/nas_optimization', exist_ok=True)
    os.makedirs('logs/nas_optimization', exist_ok=True)

    all_results = []

    for model in args.models:
        print(f"\n{'='*60}")
        print(f"Optimizing: {model.upper()}")
        print(f"{'='*60}")

        for method in args.methods:
            print(f"\n[INFO] Optimizing {model} with {method}")

            # Create optimizer
            optimizer = QuantizationOptimizer(
                model_type=model,
                method=method,
                epochs=args.epochs,
                timeout_hours=args.timeout_hours,
                batch_multiplier=args.batch_multiplier,
                max_batch_size=args.max_batch_size,
                num_workers=args.num_workers,
                use_amp=args.use_amp
            )

            # Run optimization
            results = optimizer.optimize(
                n_trials=args.n_trials,
                n_jobs=args.n_jobs
            )

            all_results.append(results)

    # Generate summary report
    generate_nas_report(all_results)

    return all_results


def generate_nas_report(results):
    """Generate comprehensive NAS optimization report"""

    print("\n" + "="*80)
    print("NAS OPTIMIZATION SUMMARY")
    print("="*80)

    # Create summary table
    summary_data = []

    for result in results:
        summary_data.append({
            'Model': result['model'].upper(),
            'Method': result['method'].upper(),
            'Best Metric': f"{result['best_value']:.4f}",
            'Best LR': f"{result['best_params'].get('learning_rate', 0):.6f}",
            'Best BS': result['best_params'].get('batch_size', 0),
            'Trials': result['n_trials']
        })

    # Convert to DataFrame and display
    df = pd.DataFrame(summary_data)
    print("\n" + tabulate(df, headers='keys', tablefmt='grid', showindex=False))

    # Save detailed report
    report_path = 'results/nas_optimization/NAS_REPORT.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("NAS-BASED HYPERPARAMETER OPTIMIZATION REPORT\n")
        f.write("="*80 + "\n\n")

        for result in results:
            f.write(f"\n{result['model'].upper()} - {result['method'].upper()}\n")
            f.write("-"*40 + "\n")
            f.write(f"Best Metric: {result['best_value']:.4f}\n")
            f.write(f"Best Parameters:\n")

            for param, value in result['best_params'].items():
                if isinstance(value, float):
                    f.write(f"  {param}: {value:.6f}\n")
                else:
                    f.write(f"  {param}: {value}\n")

            f.write(f"Total Trials: {result['n_trials']}\n")

    print(f"\n[SUCCESS] Report saved to: {report_path}")

    # Create comparison plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        models = ['lstm', 'espcn', 'sasrec']
        for idx, model in enumerate(models):
            model_results = [r for r in results if r['model'] == model]

            if model_results:
                methods = [r['method'] for r in model_results]
                values = [r['best_value'] for r in model_results]

                ax = axes[idx]
                bars = ax.bar(methods, values)

                # Color best bar differently
                best_idx = np.argmax(values)
                bars[best_idx].set_color('green')

                ax.set_title(f'{model.upper()} - Best Metrics')
                ax.set_xlabel('Method')
                ax.set_ylabel('Metric Value')
                ax.set_xticklabels(methods, rotation=45)

        plt.tight_layout()
        plt.savefig('results/nas_optimization/comparison.png', dpi=300, bbox_inches='tight')
        print("[INFO] Comparison plot saved to: results/nas_optimization/comparison.png")

    except ImportError:
        print("[WARNING] matplotlib not available, skipping plot generation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NAS-based hyperparameter optimization')

    parser.add_argument(
        '--models',
        nargs='+',
        default=['lstm', 'espcn', 'sasrec'],
        help='Models to optimize'
    )

    parser.add_argument(
        '--methods',
        nargs='+',
        default=['lsq', 'pact', 'adaround', 'apot', 'dsq'],
        help='Quantization methods to optimize'
    )

    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of trials per model-method pair'
    )

    parser.add_argument(
        '--n_jobs',
        type=int,
        default=1,
        help='Number of parallel jobs'
    )

    parser.add_argument(
        '--batch_multiplier',
        type=float,
        default=1.0,
        help='Multiply sampled batch sizes by this factor to better utilize GPU memory'
    )

    parser.add_argument(
        '--max_batch_size',
        type=int,
        default=None,
        help='Optional cap on the scaled batch size'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of DataLoader workers to pass to training scripts'
    )

    parser.add_argument(
        '--use_amp',
        action='store_true',
        help='Enable automatic mixed precision during training'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs per trial'
    )

    parser.add_argument(
        '--timeout_hours',
        type=float,
        default=2.0,
        help='Timeout for each trial in hours'
    )

    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Quick test with minimal trials'
    )

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        args.models = ['lstm']
        args.methods = ['lsq']
        args.n_trials = 3
        args.epochs = 5
        print("[INFO] Running in quick test mode")

    # Install optuna if not available
    try:
        import optuna
    except ImportError:
        print("[INFO] Installing Optuna...")
        subprocess.run(['pip', 'install', 'optuna'], check=True)
        import optuna

    # Run optimization
    results = run_nas_optimization(args)

    print("\n🎯 NAS optimization completed!")
