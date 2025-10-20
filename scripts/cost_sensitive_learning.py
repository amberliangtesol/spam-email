#!/usr/bin/env python3
"""
Implement cost-sensitive learning for spam classification.
Part of Phase 3: Precision Recovery
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, make_scorer
)
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CostSensitiveClassifier:
    """Cost-sensitive spam classifier that optimizes for business costs"""
    
    def __init__(self, config_path='configs/precision_optimized_config.json'):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.cost_matrix = None
        self.results = []
        self.best_model = None
        
    def _load_config(self, config_path):
        """Load configuration file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def set_cost_matrix(self, fp_cost=1.0, fn_cost=0.3):
        """
        Set the cost matrix for misclassifications.
        
        Args:
            fp_cost: Cost of false positive (legitimate email marked as spam)
            fn_cost: Cost of false negative (spam reaches inbox)
        """
        self.cost_matrix = {
            'false_positive': fp_cost,
            'false_negative': fn_cost,
            'true_positive': 0.0,
            'true_negative': 0.0
        }
        logger.info(f"Cost matrix set - FP cost: {fp_cost}, FN cost: {fn_cost}")
    
    def calculate_total_cost(self, y_true, y_pred):
        """Calculate total misclassification cost"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        total_cost = (
            fp * self.cost_matrix['false_positive'] +
            fn * self.cost_matrix['false_negative']
        )
        
        return {
            'total_cost': total_cost,
            'fp_cost': fp * self.cost_matrix['false_positive'],
            'fn_cost': fn * self.cost_matrix['false_negative'],
            'avg_cost_per_sample': total_cost / len(y_true),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'true_negatives': int(tn)
        }
    
    def cost_scorer(self, y_true, y_pred):
        """Custom scorer that returns negative cost (for maximization)"""
        cost_result = self.calculate_total_cost(y_true, y_pred)
        return -cost_result['total_cost']
    
    def load_data(self):
        """Load and preprocess data"""
        logger.info("Loading dataset...")
        
        # Load processed data if available
        processed_path = Path(self.config['paths']['processed_data'])
        if processed_path.exists():
            df = pd.read_csv(processed_path)
            logger.info(f"Loaded processed data: {len(df)} samples")
        else:
            # Load raw data
            raw_path = Path(self.config['paths']['raw_data'])
            if not raw_path.exists():
                raise FileNotFoundError(f"Dataset not found: {raw_path}")
            
            df = pd.read_csv(raw_path, header=None, names=['label', 'text'])
            logger.info(f"Loaded raw data: {len(df)} samples")
        
        # Convert labels
        if 'label' in df.columns and df['label'].dtype == 'object':
            df['label'] = (df['label'] == 'spam').astype(int)
        
        return df
    
    def optimize_for_cost(self, X_train, X_test, y_train, y_test):
        """Find optimal model configuration for minimum cost"""
        logger.info("\nOptimizing for minimum misclassification cost...")
        
        # Test different class weight configurations
        class_weight_configs = [
            {'name': 'balanced', 'weights': 'balanced'},
            {'name': 'equal', 'weights': {0: 1.0, 1: 1.0}},
            {'name': 'spam_emphasis_1.5x', 'weights': {0: 1.0, 1: 1.5}},
            {'name': 'spam_emphasis_2x', 'weights': {0: 1.0, 1: 2.0}},
            {'name': 'spam_emphasis_3x', 'weights': {0: 1.0, 1: 3.0}},
            {'name': 'ham_emphasis_1.5x', 'weights': {0: 1.5, 1: 1.0}},
            {'name': 'ham_emphasis_2x', 'weights': {0: 2.0, 1: 1.0}},
            {'name': 'cost_based', 'weights': self._calculate_cost_based_weights()}
        ]
        
        # Test different C values
        C_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        results = []
        best_cost = float('inf')
        best_config = None
        
        for class_config in class_weight_configs:
            for C in C_values:
                logger.info(f"Testing {class_config['name']} with C={C}...")
                
                # Create pipeline
                pipeline = Pipeline([
                    ('vectorizer', TfidfVectorizer(
                        **self.config['vectorizer']
                    )),
                    ('classifier', LogisticRegression(
                        C=C,
                        class_weight=class_config['weights'],
                        random_state=42,
                        solver='liblinear',
                        max_iter=3000
                    ))
                ])
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Evaluate on test set
                y_pred = pipeline.predict(X_test)
                
                # Calculate costs
                cost_result = self.calculate_total_cost(y_test, y_pred)
                
                # Calculate standard metrics
                metrics = {
                    'config_name': class_config['name'],
                    'C': C,
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred),
                    **cost_result
                }
                
                results.append(metrics)
                
                # Check if this is the best configuration
                if cost_result['total_cost'] < best_cost:
                    best_cost = cost_result['total_cost']
                    best_config = metrics
                    self.best_model = pipeline
                
                logger.info(f"  Total cost: ${cost_result['total_cost']:.2f}")
                logger.info(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        
        self.results = results
        return results, best_config
    
    def _calculate_cost_based_weights(self):
        """Calculate class weights based on cost matrix"""
        # Weight inversely proportional to cost
        # Higher weight for class with higher misclassification cost
        fp_cost = self.cost_matrix['false_positive']
        fn_cost = self.cost_matrix['false_negative']
        
        # Calculate relative weights
        total_cost = fp_cost + fn_cost
        weight_0 = fp_cost / total_cost * 2  # Weight for ham (class 0)
        weight_1 = fn_cost / total_cost * 2  # Weight for spam (class 1)
        
        return {0: weight_0, 1: weight_1}
    
    def analyze_cost_vs_threshold(self, X_test, y_test):
        """Analyze how cost changes with decision threshold"""
        logger.info("\nAnalyzing cost vs threshold...")
        
        if not hasattr(self.best_model, 'predict_proba'):
            logger.warning("Model doesn't support probability predictions")
            return None
        
        y_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        thresholds = np.arange(0.1, 0.91, 0.05)
        threshold_results = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            cost_result = self.calculate_total_cost(y_test, y_pred)
            
            threshold_results.append({
                'threshold': threshold,
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                **cost_result
            })
        
        # Find optimal threshold for minimum cost
        optimal_idx = np.argmin([r['total_cost'] for r in threshold_results])
        optimal_threshold = threshold_results[optimal_idx]
        
        logger.info(f"Optimal threshold for minimum cost: {optimal_threshold['threshold']:.2f}")
        logger.info(f"  Total cost: ${optimal_threshold['total_cost']:.2f}")
        logger.info(f"  Precision: {optimal_threshold['precision']:.4f}")
        logger.info(f"  Recall: {optimal_threshold['recall']:.4f}")
        
        return threshold_results, optimal_threshold
    
    def compare_cost_scenarios(self, X_test, y_test):
        """Compare different cost scenarios"""
        logger.info("\nComparing different cost scenarios...")
        
        scenarios = [
            {'name': 'Equal costs', 'fp_cost': 1.0, 'fn_cost': 1.0},
            {'name': 'FP more expensive', 'fp_cost': 1.0, 'fn_cost': 0.3},
            {'name': 'FN more expensive', 'fp_cost': 0.3, 'fn_cost': 1.0},
            {'name': 'High FP cost', 'fp_cost': 5.0, 'fn_cost': 1.0},
            {'name': 'High FN cost', 'fp_cost': 1.0, 'fn_cost': 5.0}
        ]
        
        scenario_results = []
        
        for scenario in scenarios:
            # Update cost matrix
            self.set_cost_matrix(scenario['fp_cost'], scenario['fn_cost'])
            
            # Calculate cost with current model
            y_pred = self.best_model.predict(X_test)
            cost_result = self.calculate_total_cost(y_test, y_pred)
            
            scenario_results.append({
                'scenario': scenario['name'],
                'fp_cost': scenario['fp_cost'],
                'fn_cost': scenario['fn_cost'],
                **cost_result
            })
            
            logger.info(f"\n{scenario['name']}:")
            logger.info(f"  FP cost: ${scenario['fp_cost']}, FN cost: ${scenario['fn_cost']}")
            logger.info(f"  Total cost: ${cost_result['total_cost']:.2f}")
            logger.info(f"  FP: {cost_result['false_positives']}, FN: {cost_result['false_negatives']}")
        
        return scenario_results
    
    def run_analysis(self):
        """Run complete cost-sensitive analysis"""
        # Set default cost matrix
        self.set_cost_matrix(fp_cost=1.0, fn_cost=0.3)
        
        # Load data
        df = self.load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'],
            test_size=self.config['validation']['test_size'],
            random_state=self.config['validation']['random_state'],
            stratify=df['label'] if self.config['validation']['stratify'] else None
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Optimize for cost
        results, best_config = self.optimize_for_cost(X_train, X_test, y_train, y_test)
        
        logger.info(f"\nBest configuration for minimum cost:")
        logger.info(f"  Config: {best_config['config_name']}")
        logger.info(f"  C: {best_config['C']}")
        logger.info(f"  Total cost: ${best_config['total_cost']:.2f}")
        logger.info(f"  Precision: {best_config['precision']:.4f}")
        logger.info(f"  Recall: {best_config['recall']:.4f}")
        
        # Analyze cost vs threshold
        threshold_results, optimal_threshold = self.analyze_cost_vs_threshold(X_test, y_test)
        
        # Compare cost scenarios
        scenario_results = self.compare_cost_scenarios(X_test, y_test)
        
        # Calculate baseline cost (no classifier, all marked as ham)
        y_baseline = np.zeros_like(y_test)
        baseline_cost = self.calculate_total_cost(y_test, y_baseline)
        
        # Calculate cost reduction
        cost_reduction = (baseline_cost['total_cost'] - best_config['total_cost']) / baseline_cost['total_cost'] * 100
        
        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'cost_matrix': self.cost_matrix,
            'optimization_results': results,
            'best_configuration': best_config,
            'threshold_analysis': threshold_results if threshold_results else None,
            'optimal_threshold': optimal_threshold if optimal_threshold else None,
            'scenario_comparison': scenario_results,
            'baseline_cost': baseline_cost,
            'cost_reduction_percentage': cost_reduction,
            'recommendations': self.generate_recommendations(best_config, optimal_threshold)
        }
        
        reports_dir = Path(self.config['paths']['reports_dir'])
        reports_dir.mkdir(exist_ok=True)
        
        output_file = reports_dir / 'cost_sensitive_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"\nResults saved to {output_file}")
        
        # Save cost-optimized model
        if self.best_model:
            model_dir = Path(self.config['paths']['model_dir'])
            model_dir.mkdir(exist_ok=True)
            
            model_path = model_dir / f'cost_optimized_model_{datetime.now():%Y%m%d_%H%M%S}.pkl'
            joblib.dump(self.best_model, model_path)
            logger.info(f"Cost-optimized model saved to {model_path}")
        
        return output
    
    def generate_recommendations(self, best_config, optimal_threshold):
        """Generate recommendations based on cost analysis"""
        recommendations = []
        
        # Check if precision and recall targets are met
        if best_config['precision'] >= 0.90 and best_config['recall'] >= 0.93:
            recommendations.append(
                f"Current configuration meets both precision (≥0.90) and recall (≥0.93) targets "
                f"while minimizing cost"
            )
        elif best_config['precision'] >= 0.90:
            recommendations.append(
                f"Configuration optimizes for precision ({best_config['precision']:.3f}) "
                f"but recall ({best_config['recall']:.3f}) is below target"
            )
        elif best_config['recall'] >= 0.93:
            recommendations.append(
                f"Configuration optimizes for recall ({best_config['recall']:.3f}) "
                f"but precision ({best_config['precision']:.3f}) is below target"
            )
        
        # Threshold recommendations
        if optimal_threshold:
            if optimal_threshold['threshold'] != 0.5:
                recommendations.append(
                    f"Use threshold {optimal_threshold['threshold']:.2f} instead of default 0.5 "
                    f"to minimize total cost"
                )
        
        # Cost-based recommendations
        if best_config['fp_cost'] > best_config['fn_cost']:
            recommendations.append(
                "False positives are more expensive - prioritize precision in production"
            )
        else:
            recommendations.append(
                "False negatives are more expensive - prioritize recall in production"
            )
        
        # Configuration recommendations
        if 'cost_based' in best_config['config_name']:
            recommendations.append(
                "Cost-based class weights provide optimal balance for your cost structure"
            )
        
        return recommendations

def main():
    """Main execution function"""
    classifier = CostSensitiveClassifier()
    results = classifier.run_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("COST-SENSITIVE ANALYSIS SUMMARY")
    print("="*60)
    
    best = results['best_configuration']
    print(f"\nBest Configuration:")
    print(f"  Name: {best['config_name']}")
    print(f"  Total Cost: ${best['total_cost']:.2f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall: {best['recall']:.4f}")
    print(f"  F1-Score: {best['f1']:.4f}")
    
    print(f"\nCost Breakdown:")
    print(f"  False Positive Cost: ${best['fp_cost']:.2f} ({best['false_positives']} FPs)")
    print(f"  False Negative Cost: ${best['fn_cost']:.2f} ({best['false_negatives']} FNs)")
    
    print(f"\nCost Reduction:")
    print(f"  Baseline Cost: ${results['baseline_cost']['total_cost']:.2f}")
    print(f"  Optimized Cost: ${best['total_cost']:.2f}")
    print(f"  Reduction: {results['cost_reduction_percentage']:.1f}%")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    for rec in results['recommendations']:
        print(f"• {rec}")

if __name__ == '__main__':
    main()