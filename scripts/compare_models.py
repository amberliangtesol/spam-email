#!/usr/bin/env python3
"""Model comparison script for evaluating different model versions."""
import sys
import json
import joblib
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from utils.config import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare performance of different model versions."""
    
    def __init__(self):
        """Initialize model comparator."""
        config_loader = ConfigLoader()
        self.config = config_loader.config
        self.models = {}
        self.results = {}
        
    def load_models(self):
        """Load all available models from model directory."""
        model_dir = Path(self.config['paths']['model_dir'])
        
        if not model_dir.exists():
            logger.error(f"Model directory not found: {model_dir}")
            return
        
        # Look for different model versions
        model_files = {
            'baseline': model_dir / "spam_classifier_v1.0.0.pkl",
            'recall_optimized': model_dir / "spam_classifier_recall_optimized.pkl",
            'v2.0.0': model_dir / "spam_classifier_v2.0.0.pkl"
        }
        
        for name, path in model_files.items():
            if path.exists():
                try:
                    self.models[name] = joblib.load(path)
                    logger.info(f"Loaded model: {name} from {path}")
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")
        
        if not self.models:
            logger.error("No models found to compare")
        else:
            logger.info(f"Loaded {len(self.models)} models for comparison")
    
    def load_test_data(self):
        """Load test data for evaluation."""
        # Try processed data first
        data_path = self.config['paths']['processed_data']
        if not Path(data_path).exists():
            data_path = self.config['paths']['raw_data']
            df = pd.read_csv(data_path, header=None, names=['label', 'text'])
        else:
            df = pd.read_csv(data_path)
        
        # Binary labels
        df['label_binary'] = (df['label'] == 'spam').astype(int)
        
        # Split data (same seed as training)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label_binary'],
            test_size=0.2,
            random_state=42,
            stratify=df['label_binary']
        )
        
        logger.info(f"Loaded {len(X_test)} test samples")
        
        return X_test, y_test
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model.
        
        Args:
            model: Trained model
            X_test: Test data
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'model_name': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            if y_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
            
            # Calculate additional metrics
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            metrics['true_positives'] = int(tp)
            metrics['false_positives'] = int(fp)
            metrics['true_negatives'] = int(tn)
            metrics['false_negatives'] = int(fn)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return None
    
    def compare_all_models(self):
        """Compare all loaded models."""
        if not self.models:
            self.load_models()
        
        if not self.models:
            logger.error("No models available for comparison")
            return None
        
        # Load test data
        X_test, y_test = self.load_test_data()
        
        # Evaluate each model
        for name, model in self.models.items():
            logger.info(f"Evaluating {name}...")
            metrics = self.evaluate_model(model, X_test, y_test, name)
            if metrics:
                self.results[name] = metrics
        
        return self.results
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        if not self.results:
            self.compare_all_models()
        
        if not self.results:
            logger.error("No results to report")
            return None
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.results).T
        
        # Sort by F1 score
        comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        # Find best models for each metric
        best_models = {
            'accuracy': comparison_df['accuracy'].idxmax(),
            'precision': comparison_df['precision'].idxmax(),
            'recall': comparison_df['recall'].idxmax(),
            'f1_score': comparison_df['f1_score'].idxmax()
        }
        
        # Check if recall target is met
        recall_target = 0.93
        models_meeting_target = comparison_df[comparison_df['recall'] >= recall_target]
        
        report = {
            'comparison_table': comparison_df.drop('confusion_matrix', axis=1).to_dict('index'),
            'best_models': best_models,
            'recall_target': recall_target,
            'models_meeting_recall_target': models_meeting_target.index.tolist(),
            'best_overall': comparison_df.index[0],  # Best by F1
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save report
        report_path = Path(self.config['paths']['reports_dir']) / "model_comparison.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comparison report saved to {report_path}")
        
        # Save comparison table as CSV
        csv_path = Path(self.config['paths']['reports_dir']) / "model_comparison.csv"
        comparison_df.drop('confusion_matrix', axis=1).to_csv(csv_path)
        logger.info(f"Comparison table saved to {csv_path}")
        
        return report
    
    def visualize_comparison(self, save_path=None):
        """Create visualization comparing models.
        
        Args:
            save_path: Path to save visualization
        """
        if not self.results:
            self.compare_all_models()
        
        if not self.results:
            logger.error("No results to visualize")
            return
        
        # Prepare data
        df = pd.DataFrame(self.results).T
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Plot each metric
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = df[metric].values
            names = df.index
            
            # Create bar plot
            bars = ax.bar(names, values)
            
            # Color code bars
            colors = ['green' if v == max(values) else 'steelblue' for v in values]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Formatting
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add target line for recall
            if metric == 'recall':
                ax.axhline(y=0.93, color='red', linestyle='--', 
                          alpha=0.7, label='Target (0.93)')
                ax.legend()
        
        plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_confusion_matrices(self, save_path=None):
        """Plot confusion matrices for all models.
        
        Args:
            save_path: Path to save plot
        """
        if not self.results:
            self.compare_all_models()
        
        if not self.results:
            return
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            ax = axes[idx]
            cm = np.array(metrics['confusion_matrix'])
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Ham', 'Spam'],
                       yticklabels=['Ham', 'Spam'],
                       ax=ax, cbar=False)
            
            ax.set_title(f'{name}\n(F1: {metrics["f1_score"]:.3f})')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def print_comparison_summary(self):
        """Print comparison summary to console."""
        if not self.results:
            self.compare_all_models()
        
        if not self.results:
            return
        
        df = pd.DataFrame(self.results).T
        
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Print table
        print("\nPerformance Metrics:")
        print("-"*80)
        metrics_df = df[['accuracy', 'precision', 'recall', 'f1_score']].round(4)
        print(metrics_df.to_string())
        
        # Best models
        print("\n" + "-"*80)
        print("Best Models by Metric:")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            best_model = df[metric].idxmax()
            best_score = df[metric].max()
            print(f"  {metric:12} : {best_model:20} ({best_score:.4f})")
        
        # Recall target check
        print("\n" + "-"*80)
        print("Recall Target (0.93) Achievement:")
        for name, metrics in self.results.items():
            recall = metrics['recall']
            status = "✅" if recall >= 0.93 else "❌"
            print(f"  {name:20} : {recall:.4f} {status}")
        
        print("="*80)


def main():
    """Main model comparison pipeline."""
    parser = argparse.ArgumentParser(description='Compare different model versions')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--output', type=str, help='Output directory for plots')
    args = parser.parse_args()
    
    try:
        # Initialize comparator
        comparator = ModelComparator()
        
        # Generate comparison report
        report = comparator.generate_comparison_report()
        
        if report:
            # Print summary
            comparator.print_comparison_summary()
            
            # Create visualizations if requested
            if args.visualize:
                output_dir = Path(args.output or "reports")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                comparator.visualize_comparison(
                    save_path=output_dir / "model_comparison.png"
                )
                comparator.plot_confusion_matrices(
                    save_path=output_dir / "confusion_matrices.png"
                )
        
        return 0
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())