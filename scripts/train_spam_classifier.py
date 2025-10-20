#!/usr/bin/env python3
"""Training script for spam classifier model."""
import sys
import json
import joblib
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

from scripts.utils.config import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpamClassifierTrainer:
    """Trains and evaluates spam classification model."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize trainer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config_loader = ConfigLoader()
            config = config_loader.config
        
        self.config = config
        self.model = None
        self.vectorizer = None
        self.metrics = {}
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Load preprocessed data.
        
        Args:
            data_path: Path to preprocessed CSV file
            
        Returns:
            DataFrame with processed data
        """
        if data_path is None:
            data_path = self.config['paths']['processed_data']
        
        # Check if processed data exists, otherwise use raw data
        if not Path(data_path).exists():
            logger.warning(f"Processed data not found at {data_path}, using raw data")
            data_path = self.config['paths']['raw_data']
            df = pd.read_csv(data_path, header=None, names=['label', 'text'])
        else:
            df = pd.read_csv(data_path)
        
        logger.info(f"Loaded {len(df)} samples from {data_path}")
        
        # Convert labels to binary
        df['label_binary'] = (df['label'] == 'spam').astype(int)
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Split data into train and test sets.
        
        Args:
            df: DataFrame with text and labels
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        test_size = self.config['validation']['test_size']
        random_state = self.config['validation']['random_state']
        stratify = self.config['validation']['stratify']
        
        X = df['text']
        y = df['label_binary']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )
        
        logger.info(f"Split data: {len(X_train)} train, {len(X_test)} test samples")
        logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def train_vectorizer(self, X_train: pd.Series) -> TfidfVectorizer:
        """Train TF-IDF vectorizer.
        
        Args:
            X_train: Training text data
            
        Returns:
            Fitted TfidfVectorizer
        """
        vectorizer_config = self.config['vectorizer']
        
        self.vectorizer = TfidfVectorizer(
            max_features=vectorizer_config['max_features'],
            ngram_range=tuple(vectorizer_config['ngram_range']),
            min_df=vectorizer_config['min_df'],
            max_df=vectorizer_config['max_df'],
            use_idf=vectorizer_config['use_idf'],
            sublinear_tf=vectorizer_config.get('sublinear_tf', False),
            lowercase=vectorizer_config.get('lowercase', True)
        )
        
        logger.info("Training TF-IDF vectorizer...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        logger.info(f"Vectorizer trained with vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logger.info(f"Feature matrix shape: {X_train_tfidf.shape}")
        
        return X_train_tfidf
    
    def train_model(self, X_train_tfidf, y_train) -> LogisticRegression:
        """Train Logistic Regression model.
        
        Args:
            X_train_tfidf: TF-IDF transformed training data
            y_train: Training labels
            
        Returns:
            Trained LogisticRegression model
        """
        model_config = self.config['model']['params']
        
        self.model = LogisticRegression(
            C=model_config['C'],
            class_weight=model_config['class_weight'],
            random_state=model_config['random_state'],
            max_iter=model_config['max_iter'],
            solver=model_config.get('solver', 'lbfgs')
        )
        
        logger.info("Training Logistic Regression model...")
        self.model.fit(X_train_tfidf, y_train)
        
        logger.info("Model training complete")
        
        return self.model
    
    def evaluate_model(self, X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            X_test: Test text data
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Transform test data
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_tfidf)
        y_pred_proba = self.model.predict_proba(X_test_tfidf)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=['ham', 'spam'],
                                                         output_dict=True)
        }
        
        # Log metrics
        logger.info("Model Evaluation Results:")
        logger.info(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {self.metrics['precision']:.4f}")
        logger.info(f"  Recall: {self.metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {self.metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC: {self.metrics['roc_auc']:.4f}")
        
        # Check acceptance criteria
        if self.metrics['f1_score'] >= 0.92:
            logger.info("✓ F1 score meets acceptance criteria (≥ 0.92)")
        else:
            logger.warning(f"✗ F1 score below acceptance criteria: {self.metrics['f1_score']:.4f} < 0.92")
        
        return self.metrics
    
    def save_model(self, model_dir: str = None) -> Dict[str, str]:
        """Save trained model and vectorizer.
        
        Args:
            model_dir: Directory to save model files
            
        Returns:
            Dictionary with saved file paths
        """
        if model_dir is None:
            model_dir = self.config['paths']['model_dir']
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        version = self.config.get('version', '1.0.0')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = model_dir / f"spam_classifier_v{version}.pkl"
        joblib.dump(self.model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save vectorizer
        vectorizer_path = model_dir / f"tfidf_vectorizer_v{version}.pkl"
        joblib.dump(self.vectorizer, vectorizer_path)
        logger.info(f"Saved vectorizer to {vectorizer_path}")
        
        # Save configuration
        config_path = model_dir / f"config_v{version}.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Saved configuration to {config_path}")
        
        # Save metrics
        metrics_path = model_dir / f"metrics_v{version}.json"
        metrics_data = {
            'version': version,
            'timestamp': timestamp,
            'metrics': self.metrics,
            'model_params': self.config['model']['params'],
            'vectorizer_params': self.config['vectorizer']
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        return {
            'model': str(model_path),
            'vectorizer': str(vectorizer_path),
            'config': str(config_path),
            'metrics': str(metrics_path)
        }
    
    def generate_evaluation_report(self, report_dir: str = None) -> None:
        """Generate detailed evaluation report.
        
        Args:
            report_dir: Directory to save reports
        """
        if report_dir is None:
            report_dir = self.config['paths']['reports_dir']
        
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        version = self.config.get('version', '1.0.0')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save evaluation report
        report_path = report_dir / f"evaluation_v{version}.json"
        report_data = {
            'version': version,
            'timestamp': timestamp,
            'metrics': self.metrics,
            'acceptance_criteria': {
                'f1_score_target': 0.92,
                'f1_score_achieved': self.metrics['f1_score'],
                'passed': self.metrics['f1_score'] >= 0.92
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Saved evaluation report to {report_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train spam classifier model')
    parser.add_argument('--data', type=str, help='Path to training data')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--output', type=str, help='Output directory for model')
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        config_loader = ConfigLoader(args.config)
        trainer = SpamClassifierTrainer(config_loader.config)
        
        # Load data
        df = trainer.load_data(args.data)
        
        # Split data
        X_train, X_test, y_train, y_test = trainer.split_data(df)
        
        # Train vectorizer
        X_train_tfidf = trainer.train_vectorizer(X_train)
        
        # Train model
        trainer.train_model(X_train_tfidf, y_train)
        
        # Evaluate model
        metrics = trainer.evaluate_model(X_test, y_test)
        
        # Save model and reports
        saved_paths = trainer.save_model(args.output)
        trainer.generate_evaluation_report()
        
        logger.info("Training pipeline completed successfully!")
        
        return 0 if metrics['f1_score'] >= 0.92 else 1
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())