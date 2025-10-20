#!/usr/bin/env python3
"""Feature analysis script for understanding spam indicators."""
import sys
import json
import joblib
import argparse
import logging
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from utils.config import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """Analyze features that contribute to spam detection."""
    
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        """Initialize analyzer with trained model and vectorizer.
        
        Args:
            model_path: Path to trained model
            vectorizer_path: Path to trained vectorizer
        """
        config_loader = ConfigLoader()
        self.config = config_loader.config
        
        # Load model and vectorizer
        if model_path is None:
            model_dir = Path(self.config['paths']['model_dir'])
            # Try recall-optimized model first
            model_path = model_dir / "spam_classifier_recall_optimized.pkl"
            if not model_path.exists():
                model_path = model_dir / f"spam_classifier_v{self.config['version']}.pkl"
        
        if vectorizer_path is None and not Path(model_path).exists():
            vectorizer_path = Path(self.config['paths']['model_dir']) / f"tfidf_vectorizer_v{self.config['version']}.pkl"
        
        self.model = None
        self.vectorizer = None
        
        if Path(model_path).exists():
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            
            # Extract vectorizer from pipeline if present
            if hasattr(self.model, 'named_steps'):
                self.vectorizer = self.model.named_steps.get('vectorizer')
                self.classifier = self.model.named_steps.get('classifier')
            else:
                self.classifier = self.model
                if vectorizer_path and Path(vectorizer_path).exists():
                    self.vectorizer = joblib.load(vectorizer_path)
        
    def load_data(self):
        """Load and prepare data for analysis."""
        # Try processed data first
        data_path = self.config['paths']['processed_data']
        if not Path(data_path).exists():
            data_path = self.config['paths']['raw_data']
            df = pd.read_csv(data_path, header=None, names=['label', 'text'])
        else:
            df = pd.read_csv(data_path)
        
        logger.info(f"Loaded {len(df)} samples for analysis")
        
        # Binary labels
        df['label_binary'] = (df['label'] == 'spam').astype(int)
        
        return df
    
    def get_top_features(self, n_features=50):
        """Get top features by importance.
        
        Args:
            n_features: Number of top features to return
            
        Returns:
            Dictionary with spam and ham top features
        """
        if self.vectorizer is None or self.classifier is None:
            logger.error("Model or vectorizer not loaded")
            return None
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get coefficients (for logistic regression)
        if hasattr(self.classifier, 'coef_'):
            coefficients = self.classifier.coef_[0]
            
            # Sort by coefficient value
            sorted_idx = np.argsort(coefficients)
            
            # Top spam indicators (positive coefficients)
            top_spam_idx = sorted_idx[-n_features:][::-1]
            top_spam_features = [(feature_names[i], coefficients[i]) 
                               for i in top_spam_idx]
            
            # Top ham indicators (negative coefficients)
            top_ham_idx = sorted_idx[:n_features]
            top_ham_features = [(feature_names[i], coefficients[i]) 
                              for i in top_ham_idx]
            
            return {
                'spam_indicators': top_spam_features,
                'ham_indicators': top_ham_features
            }
        else:
            logger.warning("Model doesn't have coefficients")
            return None
    
    def analyze_ngrams(self, df, n=2):
        """Analyze n-gram patterns in spam vs ham.
        
        Args:
            df: DataFrame with text and labels
            n: N-gram size
            
        Returns:
            Dictionary with n-gram analysis
        """
        spam_texts = df[df['label'] == 'spam']['text'].tolist()
        ham_texts = df[df['label'] == 'ham']['text'].tolist()
        
        # Create n-gram vectorizers
        ngram_vectorizer = TfidfVectorizer(
            ngram_range=(n, n),
            max_features=100,
            lowercase=True
        )
        
        # Fit on all texts
        all_texts = spam_texts + ham_texts
        ngram_vectorizer.fit(all_texts)
        
        # Transform spam and ham separately
        spam_features = ngram_vectorizer.transform(spam_texts)
        ham_features = ngram_vectorizer.transform(ham_texts)
        
        # Get mean TF-IDF scores
        spam_mean = np.array(spam_features.mean(axis=0)).flatten()
        ham_mean = np.array(ham_features.mean(axis=0)).flatten()
        
        feature_names = ngram_vectorizer.get_feature_names_out()
        
        # Sort by difference
        diff = spam_mean - ham_mean
        sorted_idx = np.argsort(diff)
        
        # Top spam n-grams
        top_spam_ngrams = [(feature_names[i], diff[i]) 
                          for i in sorted_idx[-20:][::-1]]
        
        # Top ham n-grams
        top_ham_ngrams = [(feature_names[i], diff[i]) 
                         for i in sorted_idx[:20]]
        
        return {
            'top_spam_ngrams': top_spam_ngrams,
            'top_ham_ngrams': top_ham_ngrams,
            'ngram_size': n
        }
    
    def analyze_misclassifications(self, df):
        """Analyze patterns in misclassified samples.
        
        Args:
            df: DataFrame with text and labels
            
        Returns:
            Analysis of false positives and false negatives
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        X = df['text']
        y_true = df['label_binary']
        
        # Make predictions
        if hasattr(self.model, 'predict'):
            y_pred = self.model.predict(X)
        else:
            X_transformed = self.vectorizer.transform(X)
            y_pred = self.classifier.predict(X_transformed)
        
        # Find misclassifications
        false_positives = df[(y_true == 0) & (y_pred == 1)]
        false_negatives = df[(y_true == 1) & (y_pred == 0)]
        
        logger.info(f"False positives (ham classified as spam): {len(false_positives)}")
        logger.info(f"False negatives (spam classified as ham): {len(false_negatives)}")
        
        # Analyze patterns in false negatives (missed spam)
        fn_analysis = self._analyze_text_patterns(false_negatives['text']) if len(false_negatives) > 0 else {}
        fp_analysis = self._analyze_text_patterns(false_positives['text']) if len(false_positives) > 0 else {}
        
        return {
            'false_positives': {
                'count': len(false_positives),
                'examples': false_positives['text'].head(5).tolist() if len(false_positives) > 0 else [],
                'patterns': fp_analysis
            },
            'false_negatives': {
                'count': len(false_negatives),
                'examples': false_negatives['text'].head(5).tolist() if len(false_negatives) > 0 else [],
                'patterns': fn_analysis
            }
        }
    
    def _analyze_text_patterns(self, texts):
        """Analyze common patterns in texts.
        
        Args:
            texts: Series of text samples
            
        Returns:
            Dictionary with pattern analysis
        """
        patterns = {
            'avg_length': texts.str.len().mean(),
            'avg_words': texts.str.split().str.len().mean(),
            'contains_url': texts.str.contains(r'https?://|www\.', regex=True).mean(),
            'contains_phone': texts.str.contains(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', regex=True).mean(),
            'contains_money': texts.str.contains(r'\$|£|€|\d+\s*(dollar|pound|euro)', regex=True).mean(),
            'uppercase_ratio': texts.apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0).mean(),
            'exclamation_count': texts.str.count('!').mean()
        }
        
        return patterns
    
    def generate_feature_importance_report(self):
        """Generate comprehensive feature importance report."""
        logger.info("Generating feature importance report...")
        
        # Load data
        df = self.load_data()
        
        # Get top features
        top_features = self.get_top_features(n_features=100)
        
        # Analyze n-grams
        bigram_analysis = self.analyze_ngrams(df, n=2)
        trigram_analysis = self.analyze_ngrams(df, n=3) if len(df) > 100 else None
        
        # Analyze misclassifications
        misclass_analysis = self.analyze_misclassifications(df)
        
        # Compile report
        report = {
            'total_samples': len(df),
            'spam_ratio': (df['label'] == 'spam').mean(),
            'top_features': top_features,
            'bigram_analysis': bigram_analysis,
            'trigram_analysis': trigram_analysis,
            'misclassification_analysis': misclass_analysis,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save report
        report_path = Path(self.config['paths']['reports_dir']) / "feature_analysis.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Feature analysis report saved to {report_path}")
        
        return report
    
    def visualize_top_features(self, n_features=20, save_path=None):
        """Create visualization of top features.
        
        Args:
            n_features: Number of features to visualize
            save_path: Path to save visualization
        """
        top_features = self.get_top_features(n_features=n_features)
        
        if top_features is None:
            logger.error("Cannot visualize - no features available")
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Spam indicators
        spam_words = [f[0] for f in top_features['spam_indicators'][:n_features]]
        spam_scores = [f[1] for f in top_features['spam_indicators'][:n_features]]
        
        ax1.barh(spam_words, spam_scores, color='red', alpha=0.7)
        ax1.set_xlabel('Feature Importance')
        ax1.set_title(f'Top {n_features} Spam Indicators')
        ax1.invert_yaxis()
        
        # Ham indicators
        ham_words = [f[0] for f in top_features['ham_indicators'][:n_features]]
        ham_scores = [abs(f[1]) for f in top_features['ham_indicators'][:n_features]]
        
        ax2.barh(ham_words, ham_scores, color='green', alpha=0.7)
        ax2.set_xlabel('Feature Importance (absolute)')
        ax2.set_title(f'Top {n_features} Ham Indicators')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def print_feature_summary(self):
        """Print a summary of feature analysis."""
        top_features = self.get_top_features(n_features=10)
        
        if top_features:
            print("\n" + "="*60)
            print("TOP SPAM INDICATORS")
            print("="*60)
            for word, score in top_features['spam_indicators'][:10]:
                print(f"  {word:20} {score:+.4f}")
            
            print("\n" + "="*60)
            print("TOP HAM INDICATORS")
            print("="*60)
            for word, score in top_features['ham_indicators'][:10]:
                print(f"  {word:20} {score:+.4f}")


def main():
    """Main feature analysis pipeline."""
    parser = argparse.ArgumentParser(description='Analyze features for spam classification')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--vectorizer', type=str, help='Path to vectorizer file')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--output', type=str, help='Output path for visualizations')
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = FeatureAnalyzer(
            model_path=args.model,
            vectorizer_path=args.vectorizer
        )
        
        # Generate report
        report = analyzer.generate_feature_importance_report()
        
        # Print summary
        analyzer.print_feature_summary()
        
        # Visualize if requested
        if args.visualize:
            output_path = args.output or "reports/feature_importance.png"
            analyzer.visualize_top_features(save_path=output_path)
        
        # Print misclassification summary
        if 'misclassification_analysis' in report:
            misclass = report['misclassification_analysis']
            print(f"\nMisclassification Analysis:")
            print(f"  False Positives: {misclass['false_positives']['count']}")
            print(f"  False Negatives: {misclass['false_negatives']['count']}")
            
            if misclass['false_negatives']['count'] > 0:
                print("\n  Common patterns in missed spam:")
                for key, value in misclass['false_negatives']['patterns'].items():
                    print(f"    {key}: {value:.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Feature analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())