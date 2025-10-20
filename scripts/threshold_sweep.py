#!/usr/bin/env python3
"""Comprehensive threshold sweep for precision-recall optimization."""
import sys
import json
import joblib
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import train_test_split

from utils.config import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """Optimize decision threshold for precision-recall balance."""
    
    def __init__(self, model_path: str = None, config_path: str = None):
        """Initialize optimizer.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
        """
        config_loader = ConfigLoader(config_path or "configs/precision_optimized_config.json")
        self.config = config_loader.config
        
        # Load model
        if model_path is None:
            model_dir = Path(self.config['paths']['model_dir'])
            # Try different model versions
            for model_name in ["spam_classifier_precision_optimized.pkl",
                             "spam_classifier_recall_optimized.pkl", 
                             "spam_classifier_v3.0.0.pkl",
                             "spam_classifier_v1.0.0.pkl"]:
                model_path = model_dir / model_name
                if model_path.exists():
                    break
        
        if Path(model_path).exists():
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.results = None
        self.optimal_threshold = None
        
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
            test_size=self.config['validation']['test_size'],
            random_state=self.config['validation']['random_state'],
            stratify=df['label_binary']
        )
        
        logger.info(f"Loaded {len(X_test)} test samples")
        return X_test, y_test
    
    def sweep_thresholds(self, X_test, y_test, granularity=0.01):
        """Perform comprehensive threshold sweep.
        
        Args:
            X_test: Test data
            y_test: Test labels
            granularity: Step size for threshold sweep
            
        Returns:
            DataFrame with threshold analysis results
        """
        logger.info(f"Starting threshold sweep with granularity {granularity}")
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_test)[:, 1]
        else:
            logger.error("Model doesn't support probability prediction")
            return None
        
        # Define threshold range
        thresholds = np.arange(0.1, 0.91, granularity)
        
        results = []
        for threshold in thresholds:
            # Apply threshold
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Calculate counts
            tp = ((y_test == 1) & (y_pred == 1)).sum()
            fp = ((y_test == 0) & (y_pred == 1)).sum()
            tn = ((y_test == 0) & (y_pred == 0)).sum()
            fn = ((y_test == 1) & (y_pred == 0)).sum()
            
            # Store results
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'spam_predictions': int(y_pred.sum()),
                'ham_predictions': int(len(y_pred) - y_pred.sum()),
                'meets_targets': (precision >= self.config['threshold']['precision_target'] and 
                                recall >= self.config['threshold']['recall_target'])
            })
        
        self.results = pd.DataFrame(results)
        logger.info(f"Completed sweep of {len(thresholds)} thresholds")
        
        return self.results
    
    def find_optimal_threshold(self, constraint='both'):
        """Find optimal threshold based on constraints.
        
        Args:
            constraint: 'both', 'precision', 'recall', or 'f1'
            
        Returns:
            Dictionary with optimal threshold and metrics
        """
        if self.results is None:
            logger.error("No sweep results available. Run sweep_thresholds first.")
            return None
        
        target_precision = self.config['threshold']['precision_target']
        target_recall = self.config['threshold']['recall_target']
        
        if constraint == 'both':
            # Find thresholds meeting both targets
            valid = self.results[
                (self.results['precision'] >= target_precision) &
                (self.results['recall'] >= target_recall)
            ]
            
            if len(valid) > 0:
                # Among valid, maximize F1
                optimal_idx = valid['f1_score'].idxmax()
                optimal = valid.loc[optimal_idx]
                logger.info(f"Found {len(valid)} thresholds meeting both targets")
            else:
                # No threshold meets both, find best compromise
                logger.warning("No threshold meets both targets, finding best compromise")
                # Calculate distance from target
                self.results['target_distance'] = np.sqrt(
                    (self.results['precision'] - target_precision)**2 +
                    (self.results['recall'] - target_recall)**2
                )
                # Weight by F1 score
                self.results['weighted_score'] = self.results['f1_score'] / (1 + self.results['target_distance'])
                optimal_idx = self.results['weighted_score'].idxmax()
                optimal = self.results.loc[optimal_idx]
        
        elif constraint == 'precision':
            # Maintain precision, maximize recall
            valid = self.results[self.results['precision'] >= target_precision]
            if len(valid) > 0:
                optimal_idx = valid['recall'].idxmax()
                optimal = valid.loc[optimal_idx]
            else:
                optimal_idx = self.results['precision'].idxmax()
                optimal = self.results.loc[optimal_idx]
        
        elif constraint == 'recall':
            # Maintain recall, maximize precision
            valid = self.results[self.results['recall'] >= target_recall]
            if len(valid) > 0:
                optimal_idx = valid['precision'].idxmax()
                optimal = valid.loc[optimal_idx]
            else:
                optimal_idx = self.results['recall'].idxmax()
                optimal = self.results.loc[optimal_idx]
        
        elif constraint == 'f1':
            # Simply maximize F1
            optimal_idx = self.results['f1_score'].idxmax()
            optimal = self.results.loc[optimal_idx]
        
        else:
            raise ValueError(f"Unknown constraint: {constraint}")
        
        self.optimal_threshold = {
            'threshold': optimal['threshold'],
            'precision': optimal['precision'],
            'recall': optimal['recall'],
            'f1_score': optimal['f1_score'],
            'specificity': optimal['specificity'],
            'meets_targets': optimal['meets_targets'],
            'constraint_used': constraint
        }
        
        return self.optimal_threshold
    
    def save_optimal_threshold(self):
        """Save optimal threshold to configuration."""
        if self.optimal_threshold is None:
            logger.error("No optimal threshold found")
            return None
        
        # Create threshold configuration
        threshold_config = {
            'optimal_threshold': self.optimal_threshold['threshold'],
            'expected_metrics': {
                'precision': self.optimal_threshold['precision'],
                'recall': self.optimal_threshold['recall'],
                'f1_score': self.optimal_threshold['f1_score'],
                'specificity': self.optimal_threshold['specificity']
            },
            'targets': {
                'precision': self.config['threshold']['precision_target'],
                'recall': self.config['threshold']['recall_target']
            },
            'meets_targets': self.optimal_threshold['meets_targets'],
            'optimization_method': self.optimal_threshold['constraint_used'],
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save to file
        output_path = Path(self.config['paths']['model_dir']) / "optimal_threshold.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(threshold_config, f, indent=2)
        
        logger.info(f"Saved optimal threshold configuration to {output_path}")
        
        # Also update main config
        config_path = Path("configs/optimal_threshold.json")
        with open(config_path, 'w') as f:
            json.dump(threshold_config, f, indent=2)
        
        logger.info(f"Saved optimal threshold to {config_path}")
        
        return output_path
    
    def generate_report(self):
        """Generate comprehensive threshold analysis report."""
        if self.results is None or self.optimal_threshold is None:
            logger.error("Run sweep and optimization first")
            return None
        
        # Find thresholds for different objectives
        objectives = {
            'both_targets': self.find_optimal_threshold('both'),
            'max_precision': self.find_optimal_threshold('precision'),
            'max_recall': self.find_optimal_threshold('recall'),
            'max_f1': self.find_optimal_threshold('f1')
        }
        
        # Count valid thresholds
        valid_count = self.results['meets_targets'].sum()
        
        report = {
            'sweep_summary': {
                'total_thresholds_tested': len(self.results),
                'thresholds_meeting_targets': int(valid_count),
                'percentage_valid': (valid_count / len(self.results) * 100),
                'threshold_range': [0.1, 0.9],
                'granularity': 0.01
            },
            'optimal_thresholds': objectives,
            'metric_ranges': {
                'precision': [self.results['precision'].min(), self.results['precision'].max()],
                'recall': [self.results['recall'].min(), self.results['recall'].max()],
                'f1_score': [self.results['f1_score'].min(), self.results['f1_score'].max()],
                'specificity': [self.results['specificity'].min(), self.results['specificity'].max()]
            },
            'recommended_threshold': self.optimal_threshold,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save report
        report_path = Path(self.config['paths']['reports_dir']) / "threshold_optimization_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved report to {report_path}")
        
        # Save detailed CSV
        csv_path = Path(self.config['paths']['reports_dir']) / "threshold_sweep.csv"
        self.results.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed results to {csv_path}")
        
        return report
    
    def plot_threshold_analysis(self, save_path=None):
        """Create comprehensive threshold analysis plots.
        
        Args:
            save_path: Path to save plots
        """
        if self.results is None:
            logger.error("No results to plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Metrics vs Threshold
        ax1 = axes[0, 0]
        ax1.plot(self.results['threshold'], self.results['precision'], 
                label='Precision', color='blue', linewidth=2)
        ax1.plot(self.results['threshold'], self.results['recall'], 
                label='Recall', color='red', linewidth=2)
        ax1.plot(self.results['threshold'], self.results['f1_score'], 
                label='F1 Score', color='green', linewidth=2)
        
        # Mark targets
        ax1.axhline(y=self.config['threshold']['precision_target'], 
                   color='blue', linestyle='--', alpha=0.5)
        ax1.axhline(y=self.config['threshold']['recall_target'], 
                   color='red', linestyle='--', alpha=0.5)
        
        # Mark optimal threshold
        if self.optimal_threshold:
            ax1.axvline(x=self.optimal_threshold['threshold'], 
                       color='black', linestyle=':', alpha=0.7, 
                       label=f"Optimal ({self.optimal_threshold['threshold']:.2f})")
        
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Metrics vs Decision Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0.1, 0.9])
        ax1.set_ylim([0, 1])
        
        # Plot 2: Trade-off visualization
        ax2 = axes[0, 1]
        valid_mask = self.results['meets_targets']
        
        # All points
        ax2.scatter(self.results['recall'], self.results['precision'], 
                   alpha=0.3, color='gray', label='All thresholds')
        
        # Valid points
        if valid_mask.any():
            ax2.scatter(self.results.loc[valid_mask, 'recall'], 
                      self.results.loc[valid_mask, 'precision'],
                      color='green', s=50, alpha=0.7, label='Meeting targets')
        
        # Optimal point
        if self.optimal_threshold:
            ax2.scatter(self.optimal_threshold['recall'], 
                      self.optimal_threshold['precision'],
                      color='red', s=100, marker='*', label='Optimal', zorder=5)
        
        # Target region
        ax2.axvline(x=self.config['threshold']['recall_target'], 
                   color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=self.config['threshold']['precision_target'], 
                   color='blue', linestyle='--', alpha=0.5)
        
        # Shade target region
        ax2.fill([self.config['threshold']['recall_target'], 1, 1, 
                 self.config['threshold']['recall_target']],
                [self.config['threshold']['precision_target'], 
                 self.config['threshold']['precision_target'], 1, 1],
                alpha=0.1, color='green')
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Trade-off')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        
        # Plot 3: Confusion matrix metrics
        ax3 = axes[1, 0]
        metrics = ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']
        colors = ['green', 'red', 'blue', 'orange']
        
        for metric, color in zip(metrics, colors):
            ax3.plot(self.results['threshold'], self.results[metric], 
                    label=metric.replace('_', ' ').title(), color=color, alpha=0.7)
        
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Count')
        ax3.set_title('Confusion Matrix Components')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: F1 Score and Target Achievement
        ax4 = axes[1, 1]
        ax4.plot(self.results['threshold'], self.results['f1_score'], 
                color='purple', linewidth=2, label='F1 Score')
        
        # Shade regions meeting targets
        for i in range(len(self.results) - 1):
            if self.results.iloc[i]['meets_targets']:
                ax4.axvspan(self.results.iloc[i]['threshold'], 
                          self.results.iloc[i+1]['threshold'],
                          alpha=0.2, color='green')
        
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('F1 Score and Target Achievement Regions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Threshold Optimization Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def print_summary(self):
        """Print optimization summary."""
        if self.optimal_threshold is None:
            logger.error("No optimization results to print")
            return
        
        print("\n" + "="*60)
        print("THRESHOLD OPTIMIZATION SUMMARY")
        print("="*60)
        
        print(f"\nTargets:")
        print(f"  Precision: ≥ {self.config['threshold']['precision_target']:.2f}")
        print(f"  Recall:    ≥ {self.config['threshold']['recall_target']:.2f}")
        
        print(f"\nOptimal Threshold: {self.optimal_threshold['threshold']:.3f}")
        print(f"\nAchieved Metrics:")
        print(f"  Precision:   {self.optimal_threshold['precision']:.4f}")
        print(f"  Recall:      {self.optimal_threshold['recall']:.4f}")
        print(f"  F1 Score:    {self.optimal_threshold['f1_score']:.4f}")
        print(f"  Specificity: {self.optimal_threshold['specificity']:.4f}")
        
        if self.optimal_threshold['meets_targets']:
            print(f"\n✅ BOTH TARGETS ACHIEVED!")
        else:
            print(f"\n⚠️  Targets not fully met - best compromise found")
        
        if self.results is not None:
            valid_count = self.results['meets_targets'].sum()
            print(f"\nThreshold Analysis:")
            print(f"  Total thresholds tested: {len(self.results)}")
            print(f"  Thresholds meeting targets: {valid_count}")
            print(f"  Percentage valid: {valid_count/len(self.results)*100:.1f}%")


def main():
    """Main threshold optimization pipeline."""
    parser = argparse.ArgumentParser(description='Optimize classification threshold')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--granularity', type=float, default=0.01, 
                       help='Threshold step size')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output', type=str, help='Output path for plots')
    args = parser.parse_args()
    
    try:
        # Initialize optimizer
        optimizer = ThresholdOptimizer(
            model_path=args.model,
            config_path=args.config
        )
        
        # Load test data
        X_test, y_test = optimizer.load_test_data()
        
        # Perform threshold sweep
        results = optimizer.sweep_thresholds(X_test, y_test, args.granularity)
        
        # Find optimal threshold
        optimal = optimizer.find_optimal_threshold('both')
        
        # Save optimal threshold
        optimizer.save_optimal_threshold()
        
        # Generate report
        report = optimizer.generate_report()
        
        # Print summary
        optimizer.print_summary()
        
        # Generate plots if requested
        if args.plot:
            output_path = args.output or "reports/threshold_optimization.png"
            optimizer.plot_threshold_analysis(save_path=output_path)
        
        return 0 if optimal['meets_targets'] else 1
        
    except Exception as e:
        logger.error(f"Threshold optimization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())