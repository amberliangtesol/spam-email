#!/usr/bin/env python3
"""
Compare LinearSVC with Logistic Regression for spam classification.
Part of Phase 3: Precision Recovery
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    precision_recall_curve, roc_auc_score, confusion_matrix,
    classification_report, make_scorer
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

class ModelComparator:
    """Compare LinearSVC and Logistic Regression models"""
    
    def __init__(self, config_path='configs/precision_optimized_config.json'):
        """Initialize comparator with configuration"""
        self.config = self._load_config(config_path)
        self.results = {}
        self.models = {}
        
    def _load_config(self, config_path):
        """Load configuration file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
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
        
        # Check class distribution
        class_dist = df['label'].value_counts()
        logger.info(f"Class distribution - Ham: {class_dist.get(0, 0)}, Spam: {class_dist.get(1, 0)}")
        
        return df
    
    def create_models(self):
        """Create model configurations to compare"""
        vectorizer_config = self.config['vectorizer']
        
        # Common vectorizer
        vectorizer = TfidfVectorizer(
            max_features=vectorizer_config.get('max_features', 10000),
            ngram_range=tuple(vectorizer_config.get('ngram_range', [1, 3])),
            min_df=vectorizer_config.get('min_df', 2),
            max_df=vectorizer_config.get('max_df', 0.90),
            use_idf=vectorizer_config.get('use_idf', True),
            sublinear_tf=vectorizer_config.get('sublinear_tf', True),
            lowercase=vectorizer_config.get('lowercase', True),
            strip_accents=vectorizer_config.get('strip_accents', 'unicode')
        )
        
        models = {
            'logistic_regression': {
                'name': 'Logistic Regression',
                'pipeline': Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', LogisticRegression(
                        C=2.0,
                        class_weight={0: 1.0, 1: 1.5},
                        random_state=42,
                        solver='liblinear',
                        max_iter=3000
                    ))
                ])
            },
            'linearsvc': {
                'name': 'LinearSVC',
                'pipeline': Pipeline([
                    ('vectorizer', TfidfVectorizer(**vectorizer_config)),
                    ('classifier', LinearSVC(
                        C=2.0,
                        class_weight={0: 1.0, 1: 1.5},
                        random_state=42,
                        max_iter=5000,
                        dual=False
                    ))
                ])
            },
            'linearsvc_calibrated': {
                'name': 'LinearSVC (Calibrated)',
                'pipeline': Pipeline([
                    ('vectorizer', TfidfVectorizer(**vectorizer_config)),
                    ('classifier', CalibratedClassifierCV(
                        LinearSVC(
                            C=2.0,
                            class_weight={0: 1.0, 1: 1.5},
                            random_state=42,
                            max_iter=5000,
                            dual=False
                        ),
                        cv=3
                    ))
                ])
            },
            'linearsvc_balanced': {
                'name': 'LinearSVC (Balanced)',
                'pipeline': Pipeline([
                    ('vectorizer', TfidfVectorizer(**vectorizer_config)),
                    ('classifier', LinearSVC(
                        C=1.0,
                        class_weight='balanced',
                        random_state=42,
                        max_iter=5000,
                        dual=False
                    ))
                ])
            }
        }
        
        return models
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, model_name, pipeline):
        """Train and evaluate a single model"""
        logger.info(f"\nTraining {model_name}...")
        
        # Training time
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Prediction time
        start_time = time.time()
        y_pred = pipeline.predict(X_test)
        predict_time = time.time() - start_time
        
        # Get probabilities if available
        if hasattr(pipeline, 'predict_proba'):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, 'decision_function'):
            y_scores = pipeline.decision_function(X_test)
            # Convert decision function to probability-like scores
            y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
        else:
            y_proba = None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'train_time': train_time,
            'predict_time': predict_time,
            'predictions_per_sec': len(y_test) / predict_time
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        # Cross-validation scores
        logger.info("Performing cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
        }
        
        cv_scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, 
            scoring=scoring, n_jobs=-1
        )
        
        metrics['cv_precision'] = cv_scores['test_precision'].mean()
        metrics['cv_recall'] = cv_scores['test_recall'].mean()
        metrics['cv_f1'] = cv_scores['test_f1'].mean()
        metrics['cv_precision_std'] = cv_scores['test_precision'].std()
        metrics['cv_recall_std'] = cv_scores['test_recall'].std()
        metrics['cv_f1_std'] = cv_scores['test_f1'].std()
        
        return metrics, pipeline
    
    def compare_decision_boundaries(self, models, X_test, y_test):
        """Compare decision boundaries of different models"""
        logger.info("\nComparing decision boundaries...")
        
        boundaries = {}
        for name, model_info in models.items():
            pipeline = model_info['trained_model']
            
            if hasattr(pipeline, 'decision_function'):
                scores = pipeline.decision_function(X_test)
                boundaries[name] = {
                    'min': float(scores.min()),
                    'max': float(scores.max()),
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'median': float(np.median(scores))
                }
                
                # Analyze score distribution by class
                ham_scores = scores[y_test == 0]
                spam_scores = scores[y_test == 1]
                
                boundaries[name]['ham_mean'] = float(ham_scores.mean())
                boundaries[name]['spam_mean'] = float(spam_scores.mean())
                boundaries[name]['separation'] = float(spam_scores.mean() - ham_scores.mean())
        
        return boundaries
    
    def run_comparison(self):
        """Run full model comparison"""
        logger.info("Starting LinearSVC vs Logistic Regression comparison...")
        
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
        
        # Create and train models
        models = self.create_models()
        results = {}
        
        for model_key, model_info in models.items():
            metrics, trained_model = self.train_and_evaluate(
                X_train, X_test, y_train, y_test,
                model_info['name'], model_info['pipeline']
            )
            
            model_info['trained_model'] = trained_model
            model_info['metrics'] = metrics
            results[model_key] = metrics
            
            # Log results
            logger.info(f"\n{model_info['name']} Results:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1']:.4f}")
            logger.info(f"  Training time: {metrics['train_time']:.2f}s")
            logger.info(f"  Prediction speed: {metrics['predictions_per_sec']:.0f} samples/sec")
            
            # Check if targets are met
            if metrics['precision'] >= 0.90 and metrics['recall'] >= 0.93:
                logger.info(f"  ✓ Meets target requirements!")
        
        # Compare decision boundaries
        boundaries = self.compare_decision_boundaries(models, X_test, y_test)
        
        # Find best model for each metric
        best_models = {
            'precision': max(results.keys(), key=lambda x: results[x]['precision']),
            'recall': max(results.keys(), key=lambda x: results[x]['recall']),
            'f1': max(results.keys(), key=lambda x: results[x]['f1']),
            'speed': max(results.keys(), key=lambda x: results[x]['predictions_per_sec'])
        }
        
        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'models': results,
            'decision_boundaries': boundaries,
            'best_models': best_models,
            'recommendations': self.generate_recommendations(results)
        }
        
        reports_dir = Path(self.config['paths']['reports_dir'])
        reports_dir.mkdir(exist_ok=True)
        
        output_file = reports_dir / 'linearsvc_comparison.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"\nResults saved to {output_file}")
        
        # Save best model
        best_overall = best_models['f1']
        if best_overall != 'logistic_regression':
            logger.info(f"\nSaving {models[best_overall]['name']} as alternative model...")
            model_dir = Path(self.config['paths']['model_dir'])
            model_dir.mkdir(exist_ok=True)
            
            model_path = model_dir / f'linearsvc_best_{datetime.now():%Y%m%d_%H%M%S}.pkl'
            joblib.dump(models[best_overall]['trained_model'], model_path)
            logger.info(f"Model saved to {model_path}")
        
        return output
    
    def generate_recommendations(self, results):
        """Generate recommendations based on comparison results"""
        recommendations = []
        
        # Check if LinearSVC performs better
        lr_metrics = results['logistic_regression']
        svc_metrics = results['linearsvc']
        svc_cal_metrics = results['linearsvc_calibrated']
        
        # Precision comparison
        if svc_metrics['precision'] > lr_metrics['precision']:
            recommendations.append(
                f"LinearSVC shows {(svc_metrics['precision'] - lr_metrics['precision'])*100:.1f}% "
                f"better precision than Logistic Regression"
            )
        
        # Speed comparison
        if svc_metrics['predictions_per_sec'] > lr_metrics['predictions_per_sec']:
            speed_improvement = (svc_metrics['predictions_per_sec'] / lr_metrics['predictions_per_sec'] - 1) * 100
            recommendations.append(
                f"LinearSVC is {speed_improvement:.0f}% faster at inference time"
            )
        
        # Calibrated version
        if svc_cal_metrics['precision'] >= 0.90 and svc_cal_metrics['recall'] >= 0.93:
            recommendations.append(
                "Calibrated LinearSVC meets both precision and recall targets with probability estimates"
            )
        
        # Overall recommendation
        if svc_metrics['f1'] > lr_metrics['f1'] and svc_metrics['precision'] >= 0.90:
            recommendations.append(
                "Consider using LinearSVC for production due to better overall performance"
            )
        elif lr_metrics['precision'] >= 0.90 and lr_metrics['recall'] >= 0.93:
            recommendations.append(
                "Logistic Regression meets all requirements and provides probability estimates natively"
            )
        
        return recommendations

def main():
    """Main execution function"""
    comparator = ModelComparator()
    results = comparator.run_comparison()
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    for model_key, metrics in results['models'].items():
        print(f"\n{model_key}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    for rec in results['recommendations']:
        print(f"• {rec}")

if __name__ == '__main__':
    main()