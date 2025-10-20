#!/usr/bin/env python3
"""Hyperparameter tuning script optimized for recall."""
import sys
import json
import joblib
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    recall_score, precision_score, f1_score, make_scorer,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

from utils.config import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecallOptimizer:
    """Hyperparameter tuner optimized for maximizing recall."""
    
    def __init__(self, config_path: str = None):
        """Initialize optimizer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        config_loader = ConfigLoader(config_path or "configs/recall_optimized_config.json")
        self.config = config_loader.config
        self.best_model = None
        self.best_vectorizer = None
        self.best_params = {}
        self.tuning_results = []
        
    def load_data(self):
        """Load and prepare data for tuning."""
        # Try processed data first, fallback to raw
        data_path = self.config['paths']['processed_data']
        if not Path(data_path).exists():
            data_path = self.config['paths']['raw_data']
            df = pd.read_csv(data_path, header=None, names=['label', 'text'])
        else:
            df = pd.read_csv(data_path)
        
        logger.info(f"Loaded {len(df)} samples")
        
        # Convert labels to binary
        df['label_binary'] = (df['label'] == 'spam').astype(int)
        
        return df['text'], df['label_binary']
    
    def create_param_grid(self):
        """Create parameter grid for GridSearch.
        
        Returns:
            Dictionary of parameters to search
        """
        param_grid = {
            # Model parameters
            'classifier__C': [0.01, 0.1, 0.5, 1.0, 2.0],
            'classifier__class_weight': [
                'balanced',
                {0: 1, 1: 2},  # Give more weight to spam
                {0: 1, 1: 3},  # Even more weight to spam
                {0: 1, 1: 1.5}
            ],
            'classifier__solver': ['liblinear', 'lbfgs'],
            
            # Vectorizer parameters
            'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'vectorizer__min_df': [1, 2, 3],
            'vectorizer__max_features': [5000, 8000, 10000],
            'vectorizer__sublinear_tf': [True, False]
        }
        
        return param_grid
    
    def create_pipeline(self):
        """Create sklearn pipeline for tuning.
        
        Returns:
            Pipeline with vectorizer and classifier
        """
        from sklearn.pipeline import Pipeline
        
        vectorizer = TfidfVectorizer(
            use_idf=True,
            lowercase=True,
            strip_accents='unicode',
            max_df=0.95
        )
        
        classifier = LogisticRegression(
            random_state=42,
            max_iter=2000
        )
        
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        
        return pipeline
    
    def optimize_for_recall(self, X, y):
        """Run grid search optimized for recall.
        
        Args:
            X: Text data
            y: Binary labels
            
        Returns:
            Best model and parameters
        """
        logger.info("Starting hyperparameter tuning for recall optimization...")
        
        # Create pipeline and parameter grid
        pipeline = self.create_pipeline()
        param_grid = self.create_param_grid()
        
        # Custom scoring for recall with minimum precision constraint
        def recall_with_precision_constraint(y_true, y_pred):
            """Score that prioritizes recall but maintains minimum precision."""
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            
            # Penalize if precision drops too low
            if precision < 0.5:
                return recall * 0.5
            return recall
        
        scorer = make_scorer(recall_with_precision_constraint)
        
        # Setup cross-validation
        cv = StratifiedKFold(
            n_splits=self.config['validation'].get('cv_folds', 5),
            shuffle=True,
            random_state=42
        )
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='recall',  # Optimize for recall
            n_jobs=-1,
            verbose=2,
            return_train_score=True
        )
        
        # Fit grid search
        logger.info(f"Testing {len(param_grid['classifier__C']) * len(param_grid['classifier__class_weight']) * len(param_grid['vectorizer__ngram_range']) * 2} parameter combinations...")
        grid_search.fit(X, y)
        
        # Store results
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        # Log results
        logger.info(f"Best recall score: {grid_search.best_score_:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Analyze all results
        results_df = pd.DataFrame(grid_search.cv_results_)
        self.analyze_tuning_results(results_df)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def analyze_tuning_results(self, results_df):
        """Analyze and report tuning results.
        
        Args:
            results_df: DataFrame with GridSearchCV results
        """
        # Sort by recall score
        results_df = results_df.sort_values('mean_test_score', ascending=False)
        
        # Top 5 configurations
        logger.info("\nTop 5 configurations by recall:")
        top_5 = results_df.head(5)[['params', 'mean_test_score', 'std_test_score']]
        for idx, row in top_5.iterrows():
            logger.info(f"  Recall: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
            logger.info(f"    Params: {row['params']}")
        
        # Save detailed results
        results_path = Path(self.config['paths']['reports_dir']) / "tuning_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Detailed results saved to {results_path}")
    
    def evaluate_best_model(self, X, y):
        """Evaluate the best model with detailed metrics.
        
        Args:
            X: Text data
            y: Binary labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        # Cross-validation scores
        scoring = {
            'recall': make_scorer(recall_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'f1': make_scorer(f1_score)
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        metrics = {}
        for metric_name, scorer in scoring.items():
            scores = cross_val_score(self.best_model, X, y, cv=cv, scoring=scorer)
            metrics[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
        
        # Log results
        logger.info("\n" + "="*50)
        logger.info("BEST MODEL EVALUATION")
        logger.info("="*50)
        logger.info(f"Recall: {metrics['recall']['mean']:.4f} (±{metrics['recall']['std']:.4f})")
        logger.info(f"Precision: {metrics['precision']['mean']:.4f} (±{metrics['precision']['std']:.4f})")
        logger.info(f"F1 Score: {metrics['f1']['mean']:.4f} (±{metrics['f1']['std']:.4f})")
        
        # Check if recall target met
        target_recall = self.config['tuning'].get('target_recall', 0.93)
        if metrics['recall']['mean'] >= target_recall:
            logger.info(f"✅ TARGET MET: Recall {metrics['recall']['mean']:.4f} >= {target_recall}")
        else:
            logger.warning(f"❌ TARGET NOT MET: Recall {metrics['recall']['mean']:.4f} < {target_recall}")
        
        return metrics
    
    def save_best_configuration(self):
        """Save the best configuration and model."""
        if self.best_params is None:
            raise ValueError("No best parameters found")
        
        # Extract parameters from pipeline format
        model_params = {}
        vectorizer_params = {}
        
        for key, value in self.best_params.items():
            if key.startswith('classifier__'):
                param_name = key.replace('classifier__', '')
                model_params[param_name] = value
            elif key.startswith('vectorizer__'):
                param_name = key.replace('vectorizer__', '')
                vectorizer_params[param_name] = value
        
        # Create optimized config
        optimized_config = self.config.copy()
        optimized_config['model']['params'].update(model_params)
        optimized_config['vectorizer'].update(vectorizer_params)
        optimized_config['tuning']['best_params'] = self.best_params
        optimized_config['tuning']['optimization_date'] = datetime.now().isoformat()
        
        # Save configuration
        config_path = Path(self.config['paths']['model_dir']) / "best_recall_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(optimized_config, f, indent=2)
        
        logger.info(f"Best configuration saved to {config_path}")
        
        # Save the model
        if self.best_model:
            model_path = Path(self.config['paths']['model_dir']) / "spam_classifier_recall_optimized.pkl"
            joblib.dump(self.best_model, model_path)
            logger.info(f"Best model saved to {model_path}")
        
        return config_path
    
    def generate_report(self, metrics):
        """Generate optimization report.
        
        Args:
            metrics: Evaluation metrics dictionary
        """
        report = {
            'optimization_type': 'recall',
            'target_recall': self.config['tuning'].get('target_recall', 0.93),
            'achieved_recall': metrics['recall']['mean'],
            'precision': metrics['precision']['mean'],
            'f1_score': metrics['f1']['mean'],
            'best_parameters': self.best_params,
            'timestamp': datetime.now().isoformat(),
            'config_version': self.config.get('version', '2.0.0')
        }
        
        # Save report
        report_path = Path(self.config['paths']['reports_dir']) / "recall_optimization_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Optimization report saved to {report_path}")
        
        return report


def main():
    """Main tuning pipeline."""
    parser = argparse.ArgumentParser(description='Tune hyperparameters for recall optimization')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--quick', action='store_true', help='Quick mode with reduced parameter grid')
    args = parser.parse_args()
    
    try:
        # Initialize optimizer
        optimizer = RecallOptimizer(args.config)
        
        # Load data
        X, y = optimizer.load_data()
        
        # If quick mode, reduce parameter grid
        if args.quick:
            logger.info("Running in quick mode with reduced parameter grid")
            optimizer.create_param_grid = lambda: {
                'classifier__C': [0.1, 1.0],
                'classifier__class_weight': ['balanced', {0: 1, 1: 2}],
                'classifier__solver': ['liblinear'],
                'vectorizer__ngram_range': [(1, 1), (1, 2)],
                'vectorizer__min_df': [2],
                'vectorizer__max_features': [5000],
                'vectorizer__sublinear_tf': [True]
            }
        
        # Optimize for recall
        best_model, best_params = optimizer.optimize_for_recall(X, y)
        
        # Evaluate best model
        metrics = optimizer.evaluate_best_model(X, y)
        
        # Save results
        optimizer.save_best_configuration()
        report = optimizer.generate_report(metrics)
        
        # Success check
        if metrics['recall']['mean'] >= optimizer.config['tuning'].get('target_recall', 0.93):
            logger.info("\n✅ Recall optimization successful!")
            return 0
        else:
            logger.warning("\n⚠️ Recall target not met, but best model saved")
            return 1
        
    except Exception as e:
        logger.error(f"Tuning failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())