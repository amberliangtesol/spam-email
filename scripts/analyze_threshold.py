#!/usr/bin/env python3
"""Threshold analysis tool for optimizing classification threshold."""
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
    precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score
)

from utils.config import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThresholdAnalyzer:
    """Analyze and optimize decision threshold for spam classification."""
    
    def __init__(self, model_path: str = None):
        """Initialize analyzer with trained model.
        
        Args:
            model_path: Path to trained model
        """
        config_loader = ConfigLoader()
        self.config = config_loader.config
        
        # Load model
        if model_path is None:
            model_dir = Path(self.config['paths']['model_dir'])
            # Try recall-optimized model first
            model_path = model_dir / "spam_classifier_recall_optimized.pkl"
            if not model_path.exists():
                model_path = model_dir / f"spam_classifier_v{self.config['version']}.pkl"
        
        if Path(model_path).exists():
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    
    def load_data(self):
        """Load and prepare test data."""
        # Try processed data first
        data_path = self.config['paths']['processed_data']
        if not Path(data_path).exists():
            data_path = self.config['paths']['raw_data']
            df = pd.read_csv(data_path, header=None, names=['label', 'text'])
        else:
            df = pd.read_csv(data_path)
        
        logger.info(f"Loaded {len(df)} samples")
        
        # Binary labels
        df['label_binary'] = (df['label'] == 'spam').astype(int)
        
        # Split data (use same seed as training)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label_binary'],
            test_size=0.2,
            random_state=42,
            stratify=df['label_binary']
        )
        
        return X_test, y_test
    
    def get_predictions(self, X):
        """Get probability predictions from model.
        
        Args:
            X: Text data
            
        Returns:
            Array of spam probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            # Direct prediction
            y_proba = self.model.predict_proba(X)[:, 1]
        else:
            # Pipeline or other format
            logger.error("Model doesn't support probability prediction")
            return None
        
        return y_proba
    
    def analyze_thresholds(self, X, y_true, threshold_range=None):
        """Analyze metrics across different thresholds.
        
        Args:
            X: Text data
            y_true: True labels
            threshold_range: Range of thresholds to test
            
        Returns:
            DataFrame with threshold analysis
        """
        # Get probabilities
        y_proba = self.get_predictions(X)
        
        if y_proba is None:
            return None
        
        # Define threshold range
        if threshold_range is None:
            threshold_range = np.arange(0.1, 0.91, 0.01)
        
        results = []
        for threshold in threshold_range:
            # Apply threshold
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Count predictions
            spam_count = y_pred.sum()
            ham_count = len(y_pred) - spam_count
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'spam_predictions': spam_count,
                'ham_predictions': ham_count,
                'spam_ratio': spam_count / len(y_pred)
            })
        
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def find_optimal_threshold(self, X, y_true, optimize_for='f1'):
        """Find optimal threshold for a given metric.
        
        Args:
            X: Text data
            y_true: True labels
            optimize_for: Metric to optimize ('f1', 'recall', 'precision', 'balanced')
            
        Returns:
            Optimal threshold and metrics at that threshold
        """
        results_df = self.analyze_thresholds(X, y_true)
        
        if results_df is None:
            return None
        
        if optimize_for == 'f1':
            optimal_idx = results_df['f1_score'].idxmax()
        elif optimize_for == 'recall':
            # Find threshold that maximizes recall while keeping precision > 0.5
            filtered = results_df[results_df['precision'] > 0.5]
            if len(filtered) > 0:
                optimal_idx = filtered['recall'].idxmax()
            else:
                optimal_idx = results_df['recall'].idxmax()
        elif optimize_for == 'precision':
            # Find threshold that maximizes precision while keeping recall > 0.5
            filtered = results_df[results_df['recall'] > 0.5]
            if len(filtered) > 0:
                optimal_idx = filtered['precision'].idxmax()
            else:
                optimal_idx = results_df['precision'].idxmax()
        elif optimize_for == 'balanced':
            # Minimize difference between precision and recall
            results_df['balance'] = 1 - abs(results_df['precision'] - results_df['recall'])
            results_df['combined'] = results_df['balance'] * results_df['f1_score']
            optimal_idx = results_df['combined'].idxmax()
        else:
            raise ValueError(f"Unknown optimization target: {optimize_for}")
        
        optimal_row = results_df.iloc[optimal_idx]
        
        return {
            'threshold': optimal_row['threshold'],
            'precision': optimal_row['precision'],
            'recall': optimal_row['recall'],
            'f1_score': optimal_row['f1_score'],
            'spam_ratio': optimal_row['spam_ratio'],
            'optimization_target': optimize_for
        }
    
    def generate_threshold_report(self, X, y_true):
        """Generate comprehensive threshold analysis report.
        
        Args:
            X: Text data
            y_true: True labels
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Generating threshold analysis report...")
        
        # Analyze full range
        results_df = self.analyze_thresholds(X, y_true)
        
        if results_df is None:
            logger.error("Failed to analyze thresholds")
            return None
        
        # Find optimal thresholds for different objectives
        optimal_f1 = self.find_optimal_threshold(X, y_true, 'f1')
        optimal_recall = self.find_optimal_threshold(X, y_true, 'recall')
        optimal_precision = self.find_optimal_threshold(X, y_true, 'precision')
        optimal_balanced = self.find_optimal_threshold(X, y_true, 'balanced')
        
        # Find threshold for target recall (0.93)
        target_recall = 0.93
        recall_achieved = results_df[results_df['recall'] >= target_recall]
        if len(recall_achieved) > 0:
            # Get the one with best precision
            target_threshold_idx = recall_achieved['precision'].idxmax()
            target_threshold = recall_achieved.iloc[target_threshold_idx]
        else:
            # Get closest to target
            results_df['recall_diff'] = abs(results_df['recall'] - target_recall)
            target_threshold_idx = results_df['recall_diff'].idxmin()
            target_threshold = results_df.iloc[target_threshold_idx]
        
        report = {
            'default_threshold': 0.5,
            'default_metrics': results_df[results_df['threshold'] == 0.5].iloc[0].to_dict() if len(results_df[results_df['threshold'] == 0.5]) > 0 else None,
            'optimal_thresholds': {
                'f1_optimized': optimal_f1,
                'recall_optimized': optimal_recall,
                'precision_optimized': optimal_precision,
                'balanced': optimal_balanced
            },
            'target_recall_threshold': {
                'target': target_recall,
                'threshold': target_threshold['threshold'],
                'achieved_recall': target_threshold['recall'],
                'precision': target_threshold['precision'],
                'f1_score': target_threshold['f1_score']
            },
            'full_analysis': results_df.to_dict('records')
        }
        
        # Save report
        report_path = Path(self.config['paths']['reports_dir']) / "threshold_analysis.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Threshold analysis saved to {report_path}")
        
        # Save CSV for detailed analysis
        csv_path = Path(self.config['paths']['reports_dir']) / "threshold_sweep.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Detailed threshold sweep saved to {csv_path}")
        
        return report
    
    def plot_threshold_curves(self, X, y_true, save_path=None):
        """Plot precision, recall, and F1 curves across thresholds.
        
        Args:
            X: Text data
            y_true: True labels
            save_path: Path to save plot
        """
        results_df = self.analyze_thresholds(X, y_true)
        
        if results_df is None:
            return
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Metrics vs Threshold
        ax1.plot(results_df['threshold'], results_df['precision'], 
                label='Precision', color='blue', linewidth=2)
        ax1.plot(results_df['threshold'], results_df['recall'], 
                label='Recall', color='red', linewidth=2)
        ax1.plot(results_df['threshold'], results_df['f1_score'], 
                label='F1 Score', color='green', linewidth=2)
        
        # Mark target recall
        ax1.axhline(y=0.93, color='red', linestyle='--', alpha=0.5, label='Target Recall (0.93)')
        
        # Mark optimal points
        optimal_f1 = self.find_optimal_threshold(X, y_true, 'f1')
        ax1.scatter(optimal_f1['threshold'], optimal_f1['f1_score'], 
                   color='green', s=100, zorder=5, label='Optimal F1')
        
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Metrics vs Classification Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Plot 2: Precision-Recall Curve
        y_proba = self.get_predictions(X)
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        ax2.plot(recall, precision, linewidth=2, color='purple')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        
        # Mark target recall point
        ax2.axvline(x=0.93, color='red', linestyle='--', alpha=0.5, label='Target Recall')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def print_summary(self, report):
        """Print summary of threshold analysis.
        
        Args:
            report: Threshold analysis report
        """
        print("\n" + "="*60)
        print("THRESHOLD ANALYSIS SUMMARY")
        print("="*60)
        
        if report.get('default_metrics'):
            print(f"\nDefault Threshold (0.50):")
            print(f"  Precision: {report['default_metrics']['precision']:.4f}")
            print(f"  Recall:    {report['default_metrics']['recall']:.4f}")
            print(f"  F1 Score:  {report['default_metrics']['f1_score']:.4f}")
        
        print(f"\nOptimal Thresholds:")
        for name, metrics in report['optimal_thresholds'].items():
            print(f"\n  {name.replace('_', ' ').title()}:")
            print(f"    Threshold: {metrics['threshold']:.2f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1 Score:  {metrics['f1_score']:.4f}")
        
        target = report.get('target_recall_threshold')
        if target:
            print(f"\nTarget Recall ({target['target']:.2f}) Achievement:")
            print(f"  Threshold:  {target['threshold']:.2f}")
            print(f"  Recall:     {target['achieved_recall']:.4f}")
            print(f"  Precision:  {target['precision']:.4f}")
            print(f"  F1 Score:   {target['f1_score']:.4f}")
            
            if target['achieved_recall'] >= target['target']:
                print(f"  ✅ Target recall achieved!")
            else:
                print(f"  ❌ Target recall not achieved (gap: {target['target'] - target['achieved_recall']:.4f})")


def main():
    """Main threshold analysis pipeline."""
    parser = argparse.ArgumentParser(description='Analyze classification thresholds')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output', type=str, help='Output path for plots')
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = ThresholdAnalyzer(model_path=args.model)
        
        # Load test data
        X_test, y_test = analyzer.load_data()
        
        # Generate report
        report = analyzer.generate_threshold_report(X_test, y_test)
        
        if report:
            # Print summary
            analyzer.print_summary(report)
            
            # Generate plots if requested
            if args.plot:
                output_path = args.output or "reports/threshold_curves.png"
                analyzer.plot_threshold_curves(X_test, y_test, save_path=output_path)
        
        return 0
        
    except Exception as e:
        logger.error(f"Threshold analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())