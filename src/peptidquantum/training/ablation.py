"""Ablation study framework for hyperparameter tuning"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
import itertools

import pandas as pd


@dataclass
class AblationConfig:
    """Configuration for ablation study"""
    # Parameter grid
    learning_rates: List[float] = None
    batch_sizes: List[int] = None
    hidden_dims: List[int] = None
    message_steps: List[int] = None
    num_heads: List[int] = None
    dense_units: List[int] = None
    dropouts: List[float] = None
    weight_decays: List[float] = None
    
    def __post_init__(self):
        # Default values
        if self.learning_rates is None:
            self.learning_rates = [1e-4, 5e-4, 1e-3]
        if self.batch_sizes is None:
            self.batch_sizes = [16, 32, 64]
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128]
        if self.message_steps is None:
            self.message_steps = [4, 6, 8]
        if self.num_heads is None:
            self.num_heads = [4, 8]
        if self.dense_units is None:
            self.dense_units = [256, 512, 1024]
        if self.dropouts is None:
            self.dropouts = [0.1, 0.2, 0.3]
        if self.weight_decays is None:
            self.weight_decays = [1e-6, 1e-5, 1e-4]


class AblationStudy:
    """Ablation study manager"""
    
    def __init__(self, config: AblationConfig, output_dir: str | Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def generate_experiments(self, mode: str = "grid") -> List[Dict[str, Any]]:
        """
        Generate experiment configurations
        
        Args:
            mode: "grid" for full grid search, "random" for random sampling,
                  "one_at_a_time" for ablation study
        """
        if mode == "grid":
            return self._grid_search()
        elif mode == "one_at_a_time":
            return self._one_at_a_time()
        elif mode == "random":
            return self._random_search(n_samples=50)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _grid_search(self) -> List[Dict[str, Any]]:
        """Full grid search over all parameters"""
        param_names = [
            'learning_rate', 'batch_size', 'hidden_dim', 'message_steps',
            'num_heads', 'dense_units', 'dropout', 'weight_decay'
        ]
        
        param_values = [
            self.config.learning_rates,
            self.config.batch_sizes,
            self.config.hidden_dims,
            self.config.message_steps,
            self.config.num_heads,
            self.config.dense_units,
            self.config.dropouts,
            self.config.weight_decays
        ]
        
        experiments = []
        for values in itertools.product(*param_values):
            exp = dict(zip(param_names, values))
            experiments.append(exp)
        
        return experiments
    
    def _one_at_a_time(self) -> List[Dict[str, Any]]:
        """One-at-a-time ablation study"""
        # Baseline configuration
        baseline = {
            'learning_rate': 5e-4,
            'batch_size': 32,
            'hidden_dim': 64,
            'message_steps': 6,
            'num_heads': 8,
            'dense_units': 512,
            'dropout': 0.2,
            'weight_decay': 1e-5
        }
        
        experiments = [baseline.copy()]
        
        # Vary each parameter one at a time
        param_grids = {
            'learning_rate': self.config.learning_rates,
            'batch_size': self.config.batch_sizes,
            'hidden_dim': self.config.hidden_dims,
            'message_steps': self.config.message_steps,
            'num_heads': self.config.num_heads,
            'dense_units': self.config.dense_units,
            'dropout': self.config.dropouts,
            'weight_decay': self.config.weight_decays
        }
        
        for param_name, param_values in param_grids.items():
            for value in param_values:
                if value != baseline[param_name]:
                    exp = baseline.copy()
                    exp[param_name] = value
                    experiments.append(exp)
        
        return experiments
    
    def _random_search(self, n_samples: int = 50) -> List[Dict[str, Any]]:
        """Random search sampling"""
        import random
        random.seed(42)
        
        experiments = []
        for _ in range(n_samples):
            exp = {
                'learning_rate': random.choice(self.config.learning_rates),
                'batch_size': random.choice(self.config.batch_sizes),
                'hidden_dim': random.choice(self.config.hidden_dims),
                'message_steps': random.choice(self.config.message_steps),
                'num_heads': random.choice(self.config.num_heads),
                'dense_units': random.choice(self.config.dense_units),
                'dropout': random.choice(self.config.dropouts),
                'weight_decay': random.choice(self.config.weight_decays)
            }
            experiments.append(exp)
        
        return experiments
    
    def add_result(self, experiment: Dict[str, Any], metrics: Dict[str, float]):
        """Add experiment result"""
        result = {**experiment, **metrics}
        self.results.append(result)
        
        # Save incrementally
        self.save_results()
    
    def save_results(self):
        """Save results to CSV and JSON"""
        if not self.results:
            return
        
        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "ablation_results.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = self.output_dir / "ablation_results.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary
        self._save_summary(df)
    
    def _save_summary(self, df: pd.DataFrame):
        """Generate and save summary statistics"""
        summary = {
            "total_experiments": len(df),
            "best_f1": float(df["f1@optimal"].max()),
            "best_auprc": float(df["auprc"].max()),
            "best_roc_auc": float(df["roc_auc"].max()),
            "best_config": df.loc[df["f1@optimal"].idxmax()].to_dict()
        }
        
        # Parameter importance (variance analysis)
        param_cols = ['learning_rate', 'batch_size', 'hidden_dim', 'message_steps',
                     'num_heads', 'dense_units', 'dropout', 'weight_decay']
        
        importance = {}
        for param in param_cols:
            if param in df.columns:
                grouped = df.groupby(param)["f1@optimal"].agg(['mean', 'std'])
                importance[param] = {
                    "mean_impact": float(grouped['mean'].max() - grouped['mean'].min()),
                    "std_impact": float(grouped['std'].mean())
                }
        
        summary["parameter_importance"] = importance
        
        # Save summary
        summary_path = self.output_dir / "ablation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Ablation Study Summary")
        print('='*60)
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Best F1: {summary['best_f1']:.4f}")
        print(f"Best AUPRC: {summary['best_auprc']:.4f}")
        print(f"Best ROC-AUC: {summary['best_roc_auc']:.4f}")
        print("\nBest configuration:")
        for k, v in summary['best_config'].items():
            if k in param_cols:
                print(f"  {k}: {v}")
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get best configuration based on F1 score"""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        best_idx = df["f1@optimal"].idxmax()
        return df.loc[best_idx].to_dict()
