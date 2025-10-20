#!/usr/bin/env python3
"""Generate PR and ROC curves for model performance visualization."""
import sys
import json
import joblib
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    average_precision_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

from utils.config import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceVisualizer:
    """Visualize model performance with various curves and plots."""
    
    def __init__(self, model_paths=None):
        """Initialize visualizer with models.
        
        Args:
            model_paths: List of model paths or None to load all
        """
        config_loader = ConfigLoader()
        self.config = config_loader.config
        self.models = {}
        
        # Load models
        if model_paths is None:
            # Load all available models
            model_dir = Path(self.config['paths']['model_dir'])
            model_files = {
                'Baseline (v1.0)': 'spam_classifier_v1.0.0.pkl',
                'Recall Optimized': 'spam_classifier_recall_optimized.pkl',
                'Precision Optimized': 'spam_classifier_precision_optimized.pkl',
                'Regularized': 'spam_classifier_regularized.pkl'
            }
            
            for name, filename in model_files.items():
                path = model_dir / filename
                if path.exists():
                    self.models[name] = joblib.load(path)
                    logger.info(f"Loaded model: {name}")
        else:
            for path in model_paths:
                if Path(path).exists():
                    name = Path(path).stem
                    self.models[name] = joblib.load(path)
                    logger.info(f"Loaded model from {path}")
        
        if not self.models:
            logger.warning("No models loaded")
    
    def load_test_data(self):
        """Load and prepare test data."""
        # Try processed data first
        data_path = self.config['paths']['processed_data']
        if not Path(data_path).exists():
            data_path = self.config['paths']['raw_data']
            df = pd.read_csv(data_path, header=None, names=['label', 'text'])
        else:
            df = pd.read_csv(data_path)
        
        # Binary labels
        df['label_binary'] = (df['label'] == 'spam').astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label_binary'],
            test_size=0.2,
            random_state=42,
            stratify=df['label_binary']
        )
        
        logger.info(f"Loaded {len(X_test)} test samples")
        return X_test, y_test
    
    def plot_pr_curves(self, X_test, y_test, save_path=None):
        """Plot Precision-Recall curves for all models.
        
        Args:
            X_test: Test data
            y_test: Test labels
            save_path: Path to save plot
        """
        if not self.models:
            logger.error("No models to plot")
            return None
        
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(self.models)))
        
        for (name, model), color in zip(self.models.items(), colors):
            try:
                # Get probabilities
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                else:
                    # For SVC without probability
                    y_proba = model.decision_function(X_test)
                
                # Calculate PR curve
                precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
                avg_precision = average_precision_score(y_test, y_proba)
                
                # Plot
                plt.plot(recall, precision, color=color, linewidth=2,
                        label=f'{name} (AP={avg_precision:.3f})')
                
            except Exception as e:
                logger.warning(f"Failed to plot PR curve for {name}: {e}")
                continue
        
        # Add target lines
        plt.axvline(x=0.93, color='red', linestyle='--', alpha=0.5, 
                   label='Target Recall (0.93)')
        plt.axhline(y=0.90, color='blue', linestyle='--', alpha=0.5,
                   label='Target Precision (0.90)')
        
        # Shade target region
        plt.fill([0.93, 1, 1, 0.93], [0.90, 0.90, 1, 1],
                alpha=0.1, color='green')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved PR curves to {save_path}")
        else:
            plt.show()
        
        return plt.gcf()
    
    def plot_roc_curves(self, X_test, y_test, save_path=None):
        """Plot ROC curves for all models.
        
        Args:
            X_test: Test data
            y_test: Test labels
            save_path: Path to save plot
        """
        if not self.models:
            logger.error("No models to plot")
            return None
        
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(self.models)))
        
        for (name, model), color in zip(self.models.items(), colors):
            try:
                # Get probabilities
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_proba = model.decision_function(X_test)
                
                # Calculate ROC curve
                fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                # Plot
                plt.plot(fpr, tpr, color=color, linewidth=2,
                        label=f'{name} (AUC={roc_auc:.3f})')
                
            except Exception as e:
                logger.warning(f"Failed to plot ROC curve for {name}: {e}")
                continue
        
        # Add diagonal line
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5,
                label='Random (AUC=0.500)')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved ROC curves to {save_path}")
        else:
            plt.show()
        
        return plt.gcf()
    
    def plot_confusion_matrices(self, X_test, y_test, threshold=0.5, save_path=None):
        """Plot confusion matrices for all models.
        
        Args:
            X_test: Test data
            y_test: Test labels
            threshold: Decision threshold
            save_path: Path to save plot
        """
        if not self.models:
            logger.error("No models to plot")
            return None
        
        n_models = len(self.models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(self.models.items()):
            ax = axes[idx]
            
            try:
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    y_pred = (y_proba >= threshold).astype(int)
                else:
                    y_pred = model.predict(X_test)
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Plot
                disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                             display_labels=['Ham', 'Spam'])
                disp.plot(ax=ax, cmap='Blues', values_format='d')
                
                # Calculate metrics
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                ax.set_title(f'{name}\nP={precision:.3f} R={recall:.3f} F1={f1:.3f}',
                           fontsize=10)
                
            except Exception as e:
                logger.warning(f"Failed to plot confusion matrix for {name}: {e}")
                ax.axis('off')
                continue
        
        # Hide extra axes
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Confusion Matrices (threshold={threshold:.2f})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved confusion matrices to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_threshold_impact(self, X_test, y_test, model_name=None, save_path=None):
        """Plot impact of threshold on metrics.
        
        Args:
            X_test: Test data
            y_test: Test labels
            model_name: Specific model to analyze
            save_path: Path to save plot
        """
        if model_name is None:
            # Use first available model
            if not self.models:
                logger.error("No models available")
                return None
            model_name = list(self.models.keys())[0]
        
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            logger.error("Model doesn't support probability prediction")
            return None
        
        # Calculate metrics for different thresholds
        thresholds = np.arange(0.1, 0.91, 0.05)
        metrics = {'precision': [], 'recall': [], 'f1': [], 'accuracy': []}
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
            metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(thresholds, metrics['precision'], label='Precision', marker='o')
        plt.plot(thresholds, metrics['recall'], label='Recall', marker='s')
        plt.plot(thresholds, metrics['f1'], label='F1 Score', marker='^')
        plt.axhline(y=0.90, color='blue', linestyle='--', alpha=0.3)
        plt.axhline(y=0.93, color='red', linestyle='--', alpha=0.3)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'Metrics vs Threshold - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(metrics['recall'], metrics['precision'], marker='o')
        for i, t in enumerate(thresholds[::2]):  # Label every other point
            plt.annotate(f'{t:.2f}', 
                        (metrics['recall'][i*2], metrics['precision'][i*2]),
                        fontsize=8, alpha=0.7)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Trade-off')
        plt.axvline(x=0.93, color='red', linestyle='--', alpha=0.3)
        plt.axhline(y=0.90, color='blue', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('Threshold Impact Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved threshold impact plot to {save_path}")
        else:
            plt.show()
        
        return plt.gcf()
    
    def generate_performance_report(self, X_test, y_test):
        """Generate comprehensive performance report with visualizations.
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Dictionary with performance metrics
        """
        report = {
            'models': {},
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        for name, model in self.models.items():
            try:
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    y_pred = model.predict(X_test)
                else:
                    y_pred = model.predict(X_test)
                    y_proba = None
                
                # Calculate metrics
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score, f1_score
                )
                
                model_metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0)
                }
                
                if y_proba is not None:
                    model_metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                    model_metrics['avg_precision'] = average_precision_score(y_test, y_proba)
                
                # Check if meets targets
                model_metrics['meets_targets'] = (
                    model_metrics['precision'] >= 0.90 and
                    model_metrics['recall'] >= 0.93
                )
                
                report['models'][name] = model_metrics
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {e}")
                continue
        
        # Save report
        report_path = Path(self.config['paths']['reports_dir']) / "performance_visualization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved performance report to {report_path}")
        
        return report
    
    def print_performance_summary(self, report):
        """Print performance summary.
        
        Args:
            report: Performance report dictionary
        """
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*80)
        
        # Create comparison table
        if report['models']:
            df = pd.DataFrame(report['models']).T
            df = df.round(4)
            
            print("\nPerformance Metrics:")
            print("-"*80)
            print(df.to_string())
            
            print("\n" + "-"*80)
            print("Target Achievement (P≥0.90, R≥0.93):")
            for name, metrics in report['models'].items():
                status = "✅" if metrics['meets_targets'] else "❌"
                print(f"  {name:25} : P={metrics['precision']:.3f} R={metrics['recall']:.3f} {status}")
            
            # Find best models
            print("\n" + "-"*80)
            print("Best Models by Metric:")
            for metric in ['precision', 'recall', 'f1_score']:
                if metric in df.columns:
                    best_model = df[metric].idxmax()
                    best_value = df[metric].max()
                    print(f"  {metric:12} : {best_model:25} ({best_value:.4f})")
        
        print("="*80)


def main():
    """Main visualization pipeline."""
    parser = argparse.ArgumentParser(description='Visualize model performance curves')
    parser.add_argument('--models', nargs='+', help='Model paths to visualize')
    parser.add_argument('--output-dir', type=str, default='reports', 
                       help='Directory for output plots')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Decision threshold for confusion matrices')
    args = parser.parse_args()
    
    try:
        # Initialize visualizer
        visualizer = PerformanceVisualizer(model_paths=args.models)
        
        if not visualizer.models:
            logger.error("No models loaded, cannot create visualizations")
            return 1
        
        # Load test data
        X_test, y_test = visualizer.load_test_data()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all visualizations
        logger.info("Generating PR curves...")
        visualizer.plot_pr_curves(X_test, y_test, 
                                 save_path=output_dir / "pr_curves.png")
        
        logger.info("Generating ROC curves...")
        visualizer.plot_roc_curves(X_test, y_test,
                                  save_path=output_dir / "roc_curves.png")
        
        logger.info("Generating confusion matrices...")
        visualizer.plot_confusion_matrices(X_test, y_test, args.threshold,
                                          save_path=output_dir / "confusion_matrices.png")
        
        # Generate threshold impact for best model
        if visualizer.models:
            model_name = list(visualizer.models.keys())[0]
            logger.info(f"Generating threshold impact for {model_name}...")
            visualizer.plot_threshold_impact(X_test, y_test, model_name,
                                            save_path=output_dir / "threshold_impact.png")
        
        # Generate performance report
        report = visualizer.generate_performance_report(X_test, y_test)
        
        # Print summary
        visualizer.print_performance_summary(report)
        
        logger.info(f"All visualizations saved to {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())