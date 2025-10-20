#!/usr/bin/env python3
"""Regularization tuning script for precision optimization."""
import sys
import json
import joblib
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

from utils.config import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegularizationTuner:
    """Tune regularization parameters for precision-recall balance."""
    
    def __init__(self, config_path: str = None):
        """Initialize tuner.
        
        Args:
            config_path: Path to configuration file
        """
        config_loader = ConfigLoader(config_path or "configs/precision_optimized_config.json")
        self.config = config_loader.config
        self.results = []
        self.best_model = None
        self.best_params = None
        
    def load_data(self):
        """Load and prepare data."""
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
        
        return df['text'], df['label_binary']
    
    def create_pipeline(self, model_type='logistic', penalty='l2', C=1.0):
        """Create model pipeline with specified parameters.
        
        Args:
            model_type: 'logistic' or 'svc'
            penalty: 'l1', 'l2', or 'elasticnet'
            C: Regularization strength
            
        Returns:
            Pipeline with vectorizer and model
        """
        # Vectorizer
        vectorizer = TfidfVectorizer(
            max_features=self.config['vectorizer']['max_features'],
            ngram_range=tuple(self.config['vectorizer']['ngram_range']),
            min_df=self.config['vectorizer']['min_df'],
            max_df=self.config['vectorizer']['max_df'],
            use_idf=self.config['vectorizer']['use_idf'],
            sublinear_tf=self.config['vectorizer']['sublinear_tf']
        )
        
        # Model
        if model_type == 'logistic':
            if penalty == 'elasticnet':
                model = LogisticRegression(
                    C=C,
                    penalty='elasticnet',
                    solver='saga',
                    l1_ratio=0.5,  # 50% L1, 50% L2
                    max_iter=3000,
                    random_state=42,
                    class_weight='balanced'
                )
            else:
                # For L1, use liblinear; for L2, use lbfgs
                solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
                model = LogisticRegression(
                    C=C,
                    penalty=penalty,
                    solver=solver,
                    max_iter=3000,
                    random_state=42,
                    class_weight='balanced'
                )
        
        elif model_type == 'svc':
            model = LinearSVC(
                C=C,
                penalty=penalty,
                dual=False if penalty == 'l1' else True,
                max_iter=5000,
                random_state=42,
                class_weight='balanced'
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('model', model)
        ])
        
        return pipeline
    
    def evaluate_regularization(self, X, y, C_values, penalties, model_types):
        """Evaluate different regularization configurations.
        
        Args:
            X: Text data
            y: Labels
            C_values: List of C values to test
            penalties: List of penalties to test
            model_types: List of model types to test
            
        Returns:
            DataFrame with results
        """
        logger.info("Starting regularization evaluation...")
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Custom scorers
        scorers = {
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }
        
        results = []
        total_configs = len(C_values) * len(penalties) * len(model_types)
        config_count = 0
        
        for model_type in model_types:
            for penalty in penalties:
                # Skip invalid combinations
                if model_type == 'svc' and penalty == 'elasticnet':
                    continue
                
                for C in C_values:
                    config_count += 1
                    logger.info(f"Testing {config_count}/{total_configs}: "
                              f"{model_type} - {penalty} - C={C}")
                    
                    try:
                        # Create pipeline
                        pipeline = self.create_pipeline(model_type, penalty, C)
                        
                        # Cross-validate
                        cv_results = cross_validate(
                            pipeline, X, y, cv=cv, 
                            scoring=scorers,
                            return_train_score=True
                        )
                        
                        # Store results
                        result = {
                            'model_type': model_type,
                            'penalty': penalty,
                            'C': C,
                            'precision_mean': cv_results['test_precision'].mean(),
                            'precision_std': cv_results['test_precision'].std(),
                            'recall_mean': cv_results['test_recall'].mean(),
                            'recall_std': cv_results['test_recall'].std(),
                            'f1_mean': cv_results['test_f1'].mean(),
                            'f1_std': cv_results['test_f1'].std(),
                            'train_precision': cv_results['train_precision'].mean(),
                            'train_recall': cv_results['train_recall'].mean(),
                            'train_f1': cv_results['train_f1'].mean()
                        }
                        
                        # Check if meets targets
                        result['meets_targets'] = (
                            result['precision_mean'] >= self.config['tuning']['target_precision'] and
                            result['recall_mean'] >= self.config['tuning']['target_recall']
                        )
                        
                        results.append(result)
                        
                    except Exception as e:
                        logger.warning(f"Failed configuration: {e}")
                        continue
        
        self.results = pd.DataFrame(results)
        logger.info(f"Completed evaluation of {len(results)} configurations")
        
        return self.results
    
    def find_best_configuration(self):
        """Find best regularization configuration.
        
        Returns:
            Dictionary with best configuration
        """
        if len(self.results) == 0:
            logger.error("No results available")
            return None
        
        # First, try to find configurations meeting both targets
        valid = self.results[self.results['meets_targets']]
        
        if len(valid) > 0:
            # Among valid, choose by F1 score
            best_idx = valid['f1_mean'].idxmax()
            best = valid.loc[best_idx]
            logger.info(f"Found {len(valid)} configurations meeting targets")
        else:
            # No configuration meets both targets
            logger.warning("No configuration meets both targets, finding best compromise")
            
            # Calculate weighted score
            self.results['weighted_score'] = (
                self.results['precision_mean'] * 0.4 +
                self.results['recall_mean'] * 0.4 +
                self.results['f1_mean'] * 0.2
            )
            
            best_idx = self.results['weighted_score'].idxmax()
            best = self.results.loc[best_idx]
        
        self.best_params = {
            'model_type': best['model_type'],
            'penalty': best['penalty'],
            'C': best['C'],
            'precision': best['precision_mean'],
            'recall': best['recall_mean'],
            'f1_score': best['f1_mean'],
            'meets_targets': best['meets_targets']
        }
        
        return self.best_params
    
    def analyze_regularization_effects(self):
        """Analyze effects of different regularization types."""
        if len(self.results) == 0:
            return None
        
        analysis = {}
        
        # Group by penalty type
        for penalty in self.results['penalty'].unique():
            penalty_results = self.results[self.results['penalty'] == penalty]
            
            analysis[penalty] = {
                'avg_precision': penalty_results['precision_mean'].mean(),
                'avg_recall': penalty_results['recall_mean'].mean(),
                'avg_f1': penalty_results['f1_mean'].mean(),
                'best_C': penalty_results.loc[penalty_results['f1_mean'].idxmax(), 'C'],
                'configs_meeting_targets': penalty_results['meets_targets'].sum()
            }
        
        # Compare model types
        for model_type in self.results['model_type'].unique():
            model_results = self.results[self.results['model_type'] == model_type]
            
            analysis[f"{model_type}_performance"] = {
                'avg_precision': model_results['precision_mean'].mean(),
                'avg_recall': model_results['recall_mean'].mean(),
                'avg_f1': model_results['f1_mean'].mean(),
                'configs_meeting_targets': model_results['meets_targets'].sum()
            }
        
        return analysis
    
    def train_best_model(self, X, y):
        """Train model with best configuration.
        
        Args:
            X: Training data
            y: Training labels
            
        Returns:
            Trained model
        """
        if self.best_params is None:
            logger.error("No best configuration found")
            return None
        
        logger.info(f"Training best model: {self.best_params['model_type']} "
                   f"with {self.best_params['penalty']} penalty, C={self.best_params['C']}")
        
        # Create pipeline with best params
        self.best_model = self.create_pipeline(
            self.best_params['model_type'],
            self.best_params['penalty'],
            self.best_params['C']
        )
        
        # Train on full data
        self.best_model.fit(X, y)
        
        return self.best_model
    
    def save_results(self):
        """Save tuning results and best model."""
        # Save detailed results
        results_path = Path(self.config['paths']['reports_dir']) / "regularization_analysis.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(results_path, index=False)
        logger.info(f"Saved detailed results to {results_path}")
        
        # Save analysis report
        analysis = self.analyze_regularization_effects()
        report = {
            'best_configuration': self.best_params,
            'regularization_analysis': analysis,
            'total_configurations_tested': len(self.results),
            'configurations_meeting_targets': int(self.results['meets_targets'].sum()),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        report_path = Path(self.config['paths']['reports_dir']) / "regularization_analysis.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved analysis report to {report_path}")
        
        # Save best model if trained
        if self.best_model is not None:
            model_path = Path(self.config['paths']['model_dir']) / "spam_classifier_regularized.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.best_model, model_path)
            logger.info(f"Saved best model to {model_path}")
        
        return report
    
    def plot_regularization_effects(self, save_path=None):
        """Plot regularization effects on metrics.
        
        Args:
            save_path: Path to save plot
        """
        if len(self.results) == 0:
            return None
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: C value effects for each penalty
        ax1 = axes[0, 0]
        for penalty in self.results['penalty'].unique():
            penalty_data = self.results[self.results['penalty'] == penalty]
            C_values = sorted(penalty_data['C'].unique())
            
            precisions = [penalty_data[penalty_data['C'] == c]['precision_mean'].mean() 
                         for c in C_values]
            
            ax1.plot(C_values, precisions, marker='o', label=penalty)
        
        ax1.set_xscale('log')
        ax1.set_xlabel('C (Regularization Strength)')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision vs Regularization Strength')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Recall effects
        ax2 = axes[0, 1]
        for penalty in self.results['penalty'].unique():
            penalty_data = self.results[self.results['penalty'] == penalty]
            C_values = sorted(penalty_data['C'].unique())
            
            recalls = [penalty_data[penalty_data['C'] == c]['recall_mean'].mean() 
                      for c in C_values]
            
            ax2.plot(C_values, recalls, marker='o', label=penalty)
        
        ax2.set_xscale('log')
        ax2.set_xlabel('C (Regularization Strength)')
        ax2.set_ylabel('Recall')
        ax2.set_title('Recall vs Regularization Strength')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: F1 Score comparison
        ax3 = axes[1, 0]
        penalties = self.results['penalty'].unique()
        model_types = self.results['model_type'].unique()
        
        x = np.arange(len(penalties))
        width = 0.35
        
        for i, model_type in enumerate(model_types):
            model_data = self.results[self.results['model_type'] == model_type]
            f1_scores = [model_data[model_data['penalty'] == p]['f1_mean'].max() 
                        for p in penalties if p in model_data['penalty'].values]
            
            ax3.bar(x + i*width, f1_scores[:len(x)], width, label=model_type)
        
        ax3.set_xlabel('Penalty Type')
        ax3.set_ylabel('Best F1 Score')
        ax3.set_title('Best F1 Score by Penalty Type')
        ax3.set_xticks(x + width/2)
        ax3.set_xticklabels(penalties)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Precision-Recall scatter
        ax4 = axes[1, 1]
        
        # Color by penalty type
        colors = {'l1': 'blue', 'l2': 'red', 'elasticnet': 'green'}
        
        for penalty in self.results['penalty'].unique():
            penalty_data = self.results[self.results['penalty'] == penalty]
            ax4.scatter(penalty_data['recall_mean'], penalty_data['precision_mean'],
                       alpha=0.6, label=penalty, color=colors.get(penalty, 'gray'))
        
        # Mark target region
        ax4.axvline(x=self.config['tuning']['target_recall'], 
                   color='red', linestyle='--', alpha=0.5)
        ax4.axhline(y=self.config['tuning']['target_precision'], 
                   color='blue', linestyle='--', alpha=0.5)
        
        # Mark best configuration
        if self.best_params:
            ax4.scatter(self.best_params['recall'], self.best_params['precision'],
                       color='red', s=100, marker='*', label='Best', zorder=5)
        
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall for Different Regularizations')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0.5, 1])
        ax4.set_ylim([0.5, 1])
        
        plt.suptitle('Regularization Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def print_summary(self):
        """Print regularization tuning summary."""
        if self.best_params is None:
            logger.error("No results to summarize")
            return
        
        print("\n" + "="*60)
        print("REGULARIZATION TUNING SUMMARY")
        print("="*60)
        
        print(f"\nBest Configuration:")
        print(f"  Model Type: {self.best_params['model_type']}")
        print(f"  Penalty:    {self.best_params['penalty']}")
        print(f"  C Value:    {self.best_params['C']}")
        
        print(f"\nAchieved Metrics:")
        print(f"  Precision: {self.best_params['precision']:.4f}")
        print(f"  Recall:    {self.best_params['recall']:.4f}")
        print(f"  F1 Score:  {self.best_params['f1_score']:.4f}")
        
        if self.best_params['meets_targets']:
            print(f"\n✅ Targets achieved!")
        else:
            print(f"\n⚠️  Targets not fully met")
        
        # Regularization analysis
        analysis = self.analyze_regularization_effects()
        if analysis:
            print(f"\nRegularization Effects:")
            for penalty in ['l1', 'l2', 'elasticnet']:
                if penalty in analysis:
                    print(f"\n  {penalty.upper()}:")
                    print(f"    Avg Precision: {analysis[penalty]['avg_precision']:.4f}")
                    print(f"    Avg Recall:    {analysis[penalty]['avg_recall']:.4f}")
                    print(f"    Best C:        {analysis[penalty]['best_C']}")


def main():
    """Main regularization tuning pipeline."""
    parser = argparse.ArgumentParser(description='Tune regularization for precision')
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer tests')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output', type=str, help='Output path for plots')
    args = parser.parse_args()
    
    try:
        # Initialize tuner
        tuner = RegularizationTuner(config_path=args.config)
        
        # Load data
        X, y = tuner.load_data()
        
        # Define parameter grid
        if args.quick:
            C_values = [0.1, 1.0, 10.0]
            penalties = ['l1', 'l2']
            model_types = ['logistic']
        else:
            C_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            penalties = ['l1', 'l2', 'elasticnet']
            model_types = ['logistic', 'svc']
        
        # Evaluate regularization
        results = tuner.evaluate_regularization(X, y, C_values, penalties, model_types)
        
        # Find best configuration
        best = tuner.find_best_configuration()
        
        # Train best model
        tuner.train_best_model(X, y)
        
        # Save results
        report = tuner.save_results()
        
        # Print summary
        tuner.print_summary()
        
        # Generate plots if requested
        if args.plot:
            output_path = args.output or "reports/regularization_effects.png"
            tuner.plot_regularization_effects(save_path=output_path)
        
        return 0 if best['meets_targets'] else 1
        
    except Exception as e:
        logger.error(f"Regularization tuning failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())