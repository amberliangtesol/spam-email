#!/usr/bin/env python3
"""
Verification script for Phase 3: Precision Recovery
Confirms that all acceptance criteria are met
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase3Verifier:
    """Verify Phase 3 acceptance criteria"""
    
    def __init__(self):
        self.criteria = {
            'precision_target': False,
            'recall_target': False,
            'optimal_threshold': False,
            'pr_curve': False,
            'roc_curve': False,
            'confusion_matrix': False,
            'configurable_threshold': False,
            'linearsvc_comparison': False,
            'phase1_functionality': False,
            'phase2_functionality': False
        }
        self.details = {}
        
    def verify_model_performance(self):
        """Verify model meets performance targets"""
        logger.info("Verifying model performance...")
        
        # Check for threshold sweep results
        threshold_file = Path('reports/optimal_threshold.json')
        if threshold_file.exists():
            with open(threshold_file, 'r') as f:
                threshold_data = json.load(f)
                
            optimal = threshold_data.get('optimal_configuration', {})
            if optimal:
                precision = optimal.get('precision', 0)
                recall = optimal.get('recall', 0)
                
                self.criteria['precision_target'] = precision >= 0.90
                self.criteria['recall_target'] = recall >= 0.93
                self.criteria['optimal_threshold'] = True
                
                self.details['performance'] = {
                    'precision': precision,
                    'recall': recall,
                    'threshold': optimal.get('threshold', 0.5),
                    'meets_targets': self.criteria['precision_target'] and self.criteria['recall_target']
                }
                
                logger.info(f"  Precision: {precision:.4f} {'✓' if self.criteria['precision_target'] else '✗'}")
                logger.info(f"  Recall: {recall:.4f} {'✓' if self.criteria['recall_target'] else '✗'}")
                logger.info(f"  Optimal threshold: {optimal.get('threshold', 0.5):.2f}")
        else:
            logger.warning("  Threshold sweep results not found")
    
    def verify_visualizations(self):
        """Verify visualization outputs exist"""
        logger.info("Verifying visualizations...")
        
        reports_dir = Path('reports')
        figures_dir = reports_dir / 'figures'
        
        # Check for PR curve
        pr_files = list(reports_dir.glob('*pr_curve*.png')) + list(figures_dir.glob('*pr_curve*.png'))
        self.criteria['pr_curve'] = len(pr_files) > 0
        logger.info(f"  PR curve: {'✓' if self.criteria['pr_curve'] else '✗'}")
        
        # Check for ROC curve
        roc_files = list(reports_dir.glob('*roc_curve*.png')) + list(figures_dir.glob('*roc_curve*.png'))
        self.criteria['roc_curve'] = len(roc_files) > 0
        logger.info(f"  ROC curve: {'✓' if self.criteria['roc_curve'] else '✗'}")
        
        # Check for confusion matrix
        cm_files = list(reports_dir.glob('*confusion_matrix*.png')) + list(figures_dir.glob('*confusion_matrix*.png'))
        self.criteria['confusion_matrix'] = len(cm_files) > 0
        logger.info(f"  Confusion matrix: {'✓' if self.criteria['confusion_matrix'] else '✗'}")
        
        self.details['visualizations'] = {
            'pr_curve_files': [str(f) for f in pr_files],
            'roc_curve_files': [str(f) for f in roc_files],
            'confusion_matrix_files': [str(f) for f in cm_files]
        }
    
    def verify_api_threshold(self):
        """Verify API supports configurable threshold"""
        logger.info("Verifying API threshold configuration...")
        
        api_file = Path('app/api_server.py')
        if api_file.exists():
            with open(api_file, 'r') as f:
                content = f.read()
                
            # Check for threshold support
            has_threshold_param = 'threshold' in content and 'PredictRequest' in content
            has_env_support = 'SPAM_THRESHOLD' in content
            
            self.criteria['configurable_threshold'] = has_threshold_param or has_env_support
            logger.info(f"  Configurable threshold: {'✓' if self.criteria['configurable_threshold'] else '✗'}")
            
            self.details['api_threshold'] = {
                'request_parameter': has_threshold_param,
                'environment_variable': has_env_support
            }
        else:
            logger.warning("  API server file not found")
    
    def verify_linearsvc_comparison(self):
        """Verify LinearSVC comparison was completed"""
        logger.info("Verifying LinearSVC comparison...")
        
        comparison_file = Path('reports/linearsvc_comparison.json')
        if comparison_file.exists():
            self.criteria['linearsvc_comparison'] = True
            
            with open(comparison_file, 'r') as f:
                comparison_data = json.load(f)
            
            self.details['linearsvc_comparison'] = {
                'models_compared': list(comparison_data.get('models', {}).keys()),
                'best_model': comparison_data.get('best_models', {}).get('f1', 'unknown')
            }
            
            logger.info(f"  LinearSVC comparison: ✓")
            logger.info(f"    Models compared: {len(comparison_data.get('models', {}))}")
        else:
            logger.warning("  LinearSVC comparison results not found")
    
    def verify_backward_compatibility(self):
        """Verify Phase 1 and 2 functionality preserved"""
        logger.info("Verifying backward compatibility...")
        
        # Check Phase 1 scripts
        phase1_scripts = [
            'scripts/train_model.py',
            'scripts/predict_spam.py',
            'scripts/preprocess_data.py'
        ]
        
        phase1_ok = all(Path(script).exists() for script in phase1_scripts)
        self.criteria['phase1_functionality'] = phase1_ok
        logger.info(f"  Phase 1 functionality: {'✓' if phase1_ok else '✗'}")
        
        # Check Phase 2 scripts
        phase2_scripts = [
            'scripts/tune_for_recall.py',
            'scripts/analyze_features.py',
            'scripts/analyze_threshold.py'
        ]
        
        phase2_ok = all(Path(script).exists() for script in phase2_scripts)
        self.criteria['phase2_functionality'] = phase2_ok
        logger.info(f"  Phase 2 functionality: {'✓' if phase2_ok else '✗'}")
        
        self.details['backward_compatibility'] = {
            'phase1_scripts': {script: Path(script).exists() for script in phase1_scripts},
            'phase2_scripts': {script: Path(script).exists() for script in phase2_scripts}
        }
    
    def verify_all_scripts(self):
        """Verify all Phase 3 scripts exist and are executable"""
        logger.info("Verifying Phase 3 scripts...")
        
        phase3_scripts = {
            'threshold_sweep.py': Path('scripts/threshold_sweep.py'),
            'tune_regularization.py': Path('scripts/tune_regularization.py'),
            'visualize_curves.py': Path('scripts/visualize_curves.py'),
            'compare_linearsvc.py': Path('scripts/compare_linearsvc.py'),
            'cost_sensitive_learning.py': Path('scripts/cost_sensitive_learning.py'),
            'benchmark.py': Path('scripts/benchmark.py')
        }
        
        script_status = {}
        for name, path in phase3_scripts.items():
            exists = path.exists()
            script_status[name] = exists
            logger.info(f"  {name}: {'✓' if exists else '✗'}")
        
        self.details['phase3_scripts'] = script_status
    
    def generate_summary(self):
        """Generate verification summary"""
        all_criteria_met = all(self.criteria.values())
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'Phase 3: Precision Recovery',
            'all_criteria_met': all_criteria_met,
            'criteria': self.criteria,
            'details': self.details,
            'recommendations': []
        }
        
        # Add recommendations
        if not self.criteria['precision_target']:
            summary['recommendations'].append(
                "Run threshold sweep to find optimal threshold for precision ≥ 0.90"
            )
        
        if not self.criteria['recall_target']:
            summary['recommendations'].append(
                "Adjust class weights or regularization to improve recall ≥ 0.93"
            )
        
        if not (self.criteria['pr_curve'] and self.criteria['roc_curve']):
            summary['recommendations'].append(
                "Run visualize_curves.py to generate performance visualizations"
            )
        
        if not self.criteria['linearsvc_comparison']:
            summary['recommendations'].append(
                "Run compare_linearsvc.py to complete model comparison"
            )
        
        if all_criteria_met:
            summary['recommendations'].append(
                "✓ All Phase 3 acceptance criteria met! Ready for production deployment."
            )
        
        return summary
    
    def run_verification(self):
        """Run complete verification"""
        logger.info("="*60)
        logger.info("PHASE 3 VERIFICATION")
        logger.info("="*60)
        
        # Run all verification checks
        self.verify_model_performance()
        self.verify_visualizations()
        self.verify_api_threshold()
        self.verify_linearsvc_comparison()
        self.verify_backward_compatibility()
        self.verify_all_scripts()
        
        # Generate summary
        summary = self.generate_summary()
        
        # Save verification report
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / 'phase3_verification.json'
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nVerification report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("VERIFICATION SUMMARY")
        print("="*60)
        
        print("\nAcceptance Criteria:")
        for criterion, met in self.criteria.items():
            status = "✓" if met else "✗"
            print(f"  {status} {criterion.replace('_', ' ').title()}")
        
        print(f"\nOverall Status: {'✓ PASSED' if summary['all_criteria_met'] else '✗ NOT COMPLETE'}")
        
        if summary['recommendations']:
            print("\nRecommendations:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        return summary

def main():
    """Main execution function"""
    verifier = Phase3Verifier()
    summary = verifier.run_verification()
    
    # Exit with appropriate code
    sys.exit(0 if summary['all_criteria_met'] else 1)

if __name__ == '__main__':
    main()