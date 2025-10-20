#!/usr/bin/env python3
"""
Comprehensive benchmark suite for spam classifier across all phases.
Part of Phase 3: Precision Recovery
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkSuite:
    """Comprehensive benchmark for all model phases"""
    
    def __init__(self):
        """Initialize benchmark suite"""
        self.phases = {
            'phase1': {
                'name': 'Baseline Classifier',
                'config': 'configs/default_config.json',
                'models': []
            },
            'phase2': {
                'name': 'Recall Optimized',
                'config': 'configs/recall_optimized_config.json',
                'models': []
            },
            'phase3': {
                'name': 'Precision Recovery',
                'config': 'configs/precision_optimized_config.json',
                'models': []
            }
        }
        self.results = {}
        self.best_configs = {}
        
    def find_models(self):
        """Find all trained models for each phase"""
        logger.info("Searching for trained models...")
        
        model_dir = Path('models')
        if not model_dir.exists():
            logger.warning("Models directory not found")
            return
        
        # Model patterns for each phase
        patterns = {
            'phase1': ['spam_classifier_v*.pkl', 'baseline_*.pkl'],
            'phase2': ['recall_optimized_*.pkl', 'tuned_model_*.pkl'],
            'phase3': ['precision_optimized_*.pkl', 'cost_optimized_*.pkl', 'linearsvc_best_*.pkl']
        }
        
        for phase, phase_patterns in patterns.items():
            for pattern in phase_patterns:
                for model_path in model_dir.glob(pattern):
                    self.phases[phase]['models'].append(str(model_path))
                    logger.info(f"Found {phase} model: {model_path.name}")
        
        # Add the latest model from each phase if specific ones not found
        for phase in self.phases:
            if not self.phases[phase]['models']:
                # Try to find any model with phase indicators
                if phase == 'phase1':
                    latest = list(model_dir.glob('spam_classifier*.pkl'))
                elif phase == 'phase2':
                    latest = list(model_dir.glob('*recall*.pkl'))
                else:  # phase3
                    latest = list(model_dir.glob('*precision*.pkl'))
                
                if latest:
                    # Sort by modification time and take the latest
                    latest.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    self.phases[phase]['models'].append(str(latest[0]))
                    logger.info(f"Found {phase} model: {latest[0].name}")
    
    def load_data(self, config_path):
        """Load dataset based on configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load processed data if available
        processed_path = Path(config['paths'].get('processed_data', 'datasets/processed/train_data.csv'))
        if processed_path.exists():
            df = pd.read_csv(processed_path)
        else:
            # Load raw data
            raw_path = Path(config['paths'].get('raw_data', 'datasets/sms_spam_no_header.csv'))
            if not raw_path.exists():
                raise FileNotFoundError(f"Dataset not found: {raw_path}")
            
            df = pd.read_csv(raw_path, header=None, names=['label', 'text'])
        
        # Convert labels
        if 'label' in df.columns and df['label'].dtype == 'object':
            df['label'] = (df['label'] == 'spam').astype(int)
        
        return df, config
    
    def evaluate_model(self, model_path, X_test, y_test):
        """Evaluate a single model"""
        logger.info(f"Evaluating {Path(model_path).name}...")
        
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                y_proba = None
                roc_auc = None
            
            # Calculate metrics
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            metrics = {
                'model_path': model_path,
                'model_name': Path(model_path).name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc,
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'inference_time_ms': round(inference_time, 3),
                'meets_precision_target': precision_score(y_test, y_pred) >= 0.90,
                'meets_recall_target': recall_score(y_test, y_pred) >= 0.93,
                'meets_both_targets': (
                    precision_score(y_test, y_pred) >= 0.90 and
                    recall_score(y_test, y_pred) >= 0.93
                )
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_path}: {e}")
            return None
    
    def benchmark_phase(self, phase_key, phase_info):
        """Benchmark all models in a phase"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking {phase_info['name']}")
        logger.info(f"{'='*60}")
        
        if not phase_info['models']:
            logger.warning(f"No models found for {phase_info['name']}")
            return []
        
        # Load data with phase-specific config
        df, config = self.load_data(phase_info['config'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'],
            test_size=config['validation']['test_size'],
            random_state=config['validation']['random_state'],
            stratify=df['label'] if config['validation']['stratify'] else None
        )
        
        phase_results = []
        for model_path in phase_info['models']:
            metrics = self.evaluate_model(model_path, X_test, y_test)
            if metrics:
                phase_results.append(metrics)
                
                # Log key metrics
                logger.info(f"\n{metrics['model_name']}:")
                logger.info(f"  Precision: {metrics['precision']:.4f} {'✓' if metrics['meets_precision_target'] else '✗'}")
                logger.info(f"  Recall: {metrics['recall']:.4f} {'✓' if metrics['meets_recall_target'] else '✗'}")
                logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
                logger.info(f"  Inference: {metrics['inference_time_ms']:.3f}ms/sample")
        
        return phase_results
    
    def compare_phases(self):
        """Compare best models from each phase"""
        logger.info(f"\n{'='*60}")
        logger.info("PHASE COMPARISON")
        logger.info(f"{'='*60}")
        
        comparison = []
        for phase_key, phase_results in self.results.items():
            if not phase_results:
                continue
            
            # Find best model in phase by F1 score
            best_model = max(phase_results, key=lambda x: x['f1_score'])
            
            comparison.append({
                'phase': self.phases[phase_key]['name'],
                'best_model': best_model['model_name'],
                'precision': best_model['precision'],
                'recall': best_model['recall'],
                'f1_score': best_model['f1_score'],
                'accuracy': best_model['accuracy'],
                'meets_targets': best_model['meets_both_targets'],
                'inference_time_ms': best_model['inference_time_ms']
            })
            
            self.best_configs[phase_key] = best_model
        
        # Print comparison table
        if comparison:
            df_comp = pd.DataFrame(comparison)
            print("\nBest Model from Each Phase:")
            print(df_comp.to_string(index=False))
        
        return comparison
    
    def generate_recommendations(self):
        """Generate recommendations based on benchmark results"""
        recommendations = []
        
        # Find models meeting both targets
        models_meeting_targets = []
        for phase_results in self.results.values():
            for model in phase_results:
                if model['meets_both_targets']:
                    models_meeting_targets.append(model)
        
        if models_meeting_targets:
            best_overall = max(models_meeting_targets, key=lambda x: x['f1_score'])
            recommendations.append(
                f"PRODUCTION READY: '{best_overall['model_name']}' meets both precision (≥0.90) "
                f"and recall (≥0.93) targets with F1={best_overall['f1_score']:.4f}"
            )
        else:
            recommendations.append(
                "WARNING: No models currently meet both precision (≥0.90) and recall (≥0.93) targets"
            )
        
        # Phase-specific recommendations
        if 'phase3' in self.best_configs:
            phase3_best = self.best_configs['phase3']
            if phase3_best['meets_precision_target'] and not phase3_best['meets_recall_target']:
                recommendations.append(
                    f"Phase 3 achieved precision target ({phase3_best['precision']:.4f}) "
                    f"but needs recall improvement ({phase3_best['recall']:.4f} < 0.93)"
                )
        
        # Speed recommendations
        fastest_model = None
        for phase_results in self.results.values():
            for model in phase_results:
                if fastest_model is None or model['inference_time_ms'] < fastest_model['inference_time_ms']:
                    fastest_model = model
        
        if fastest_model:
            recommendations.append(
                f"Fastest inference: '{fastest_model['model_name']}' "
                f"at {fastest_model['inference_time_ms']:.3f}ms/sample"
            )
        
        # Trade-off analysis
        high_precision = []
        high_recall = []
        for phase_results in self.results.values():
            for model in phase_results:
                if model['precision'] >= 0.95:
                    high_precision.append(model)
                if model['recall'] >= 0.95:
                    high_recall.append(model)
        
        if high_precision:
            best_precision = max(high_precision, key=lambda x: x['precision'])
            recommendations.append(
                f"Highest precision: '{best_precision['model_name']}' "
                f"with {best_precision['precision']:.4f}"
            )
        
        if high_recall:
            best_recall = max(high_recall, key=lambda x: x['recall'])
            recommendations.append(
                f"Highest recall: '{best_recall['model_name']}' "
                f"with {best_recall['recall']:.4f}"
            )
        
        return recommendations
    
    def run_benchmark(self):
        """Run complete benchmark suite"""
        logger.info("Starting comprehensive benchmark suite...")
        
        # Find all available models
        self.find_models()
        
        # Benchmark each phase
        for phase_key, phase_info in self.phases.items():
            phase_results = self.benchmark_phase(phase_key, phase_info)
            self.results[phase_key] = phase_results
        
        # Compare phases
        comparison = self.compare_phases()
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'phases': self.phases,
            'results': self.results,
            'comparison': comparison,
            'best_configs': self.best_configs,
            'recommendations': recommendations
        }
        
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        json_file = reports_dir / 'benchmark_results.json'
        with open(json_file, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"\nJSON results saved to {json_file}")
        
        # Generate Markdown report
        self.generate_markdown_report(output, reports_dir / 'benchmark_results.md')
        
        return output
    
    def generate_markdown_report(self, results, output_path):
        """Generate a markdown report of benchmark results"""
        with open(output_path, 'w') as f:
            f.write("# Spam Classifier Benchmark Results\n\n")
            f.write(f"Generated: {results['timestamp']}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            if results['recommendations']:
                for rec in results['recommendations']:
                    f.write(f"- {rec}\n")
            f.write("\n")
            
            # Phase Results
            f.write("## Phase Results\n\n")
            
            for phase_key in ['phase1', 'phase2', 'phase3']:
                if phase_key not in results['results']:
                    continue
                    
                phase_name = self.phases[phase_key]['name']
                phase_results = results['results'][phase_key]
                
                f.write(f"### {phase_name}\n\n")
                
                if not phase_results:
                    f.write("No models found for this phase.\n\n")
                    continue
                
                # Create results table
                f.write("| Model | Precision | Recall | F1-Score | Meets Targets |\n")
                f.write("|-------|-----------|--------|----------|---------------|\n")
                
                for model in phase_results:
                    meets = "✓ Both" if model['meets_both_targets'] else (
                        "✓ Precision" if model['meets_precision_target'] else (
                            "✓ Recall" if model['meets_recall_target'] else "✗ Neither"
                        )
                    )
                    f.write(f"| {model['model_name'][:30]} | "
                           f"{model['precision']:.4f} | "
                           f"{model['recall']:.4f} | "
                           f"{model['f1_score']:.4f} | "
                           f"{meets} |\n")
                f.write("\n")
            
            # Best Configuration for Each Metric
            f.write("## Best Configuration for Each Metric\n\n")
            
            all_models = []
            for phase_results in results['results'].values():
                all_models.extend(phase_results)
            
            if all_models:
                best_by_metric = {
                    'Precision': max(all_models, key=lambda x: x['precision']),
                    'Recall': max(all_models, key=lambda x: x['recall']),
                    'F1-Score': max(all_models, key=lambda x: x['f1_score']),
                    'Speed': min(all_models, key=lambda x: x['inference_time_ms'])
                }
                
                f.write("| Metric | Best Model | Value |\n")
                f.write("|--------|------------|-------|\n")
                
                for metric, model in best_by_metric.items():
                    if metric == 'Speed':
                        value = f"{model['inference_time_ms']:.3f}ms"
                    elif metric == 'Precision':
                        value = f"{model['precision']:.4f}"
                    elif metric == 'Recall':
                        value = f"{model['recall']:.4f}"
                    else:
                        value = f"{model['f1_score']:.4f}"
                    
                    f.write(f"| {metric} | {model['model_name'][:30]} | {value} |\n")
                f.write("\n")
            
            # Trade-offs Analysis
            f.write("## Trade-off Analysis\n\n")
            f.write("### Precision vs Recall\n\n")
            f.write("Models are evaluated against:\n")
            f.write("- **Precision Target**: ≥ 0.90 (minimize false positives)\n")
            f.write("- **Recall Target**: ≥ 0.93 (minimize false negatives)\n\n")
            
            # Recommendations for Production
            f.write("## Recommendations for Production\n\n")
            
            models_meeting_both = [m for m in all_models if m['meets_both_targets']]
            if models_meeting_both:
                best_production = max(models_meeting_both, key=lambda x: x['f1_score'])
                f.write(f"### Recommended Model: `{best_production['model_name']}`\n\n")
                f.write(f"- **Precision**: {best_production['precision']:.4f} ✓\n")
                f.write(f"- **Recall**: {best_production['recall']:.4f} ✓\n")
                f.write(f"- **F1-Score**: {best_production['f1_score']:.4f}\n")
                f.write(f"- **Inference Time**: {best_production['inference_time_ms']:.3f}ms/sample\n")
                f.write("\nThis model successfully balances both precision and recall requirements.\n")
            else:
                f.write("### No Model Currently Meets Both Targets\n\n")
                f.write("Consider:\n")
                f.write("1. Further threshold optimization\n")
                f.write("2. Ensemble methods combining high-precision and high-recall models\n")
                f.write("3. Additional feature engineering\n")
                f.write("4. Collecting more training data\n")
        
        logger.info(f"Markdown report saved to {output_path}")

def main():
    """Main execution function"""
    suite = BenchmarkSuite()
    results = suite.run_benchmark()
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print("\nRecommendations:")
    for rec in results['recommendations']:
        print(f"• {rec}")
    
    print(f"\nDetailed reports available in:")
    print(f"  - reports/benchmark_results.json")
    print(f"  - reports/benchmark_results.md")

if __name__ == '__main__':
    main()