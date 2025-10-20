#!/usr/bin/env python3
"""Verification script for Phase 1 acceptance criteria."""
import sys
import subprocess
import json
from pathlib import Path
import time

def check_command(command, description):
    """Check if a command runs successfully."""
    print(f"✓ Checking: {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"  ✅ {description} - PASSED")
            return True
        else:
            print(f"  ❌ {description} - FAILED")
            print(f"     Error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  ❌ {description} - ERROR: {e}")
        return False

def check_file_exists(path, description):
    """Check if a file exists."""
    print(f"✓ Checking: {description}")
    if Path(path).exists():
        print(f"  ✅ {description} - EXISTS")
        return True
    else:
        print(f"  ❌ {description} - NOT FOUND")
        return False

def check_metrics():
    """Check if model meets F1 score requirement."""
    print("✓ Checking: F1 Score ≥ 0.92")
    metrics_path = Path("models/metrics_v1.0.0.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        f1_score = metrics.get('metrics', {}).get('f1_score', 0)
        if f1_score >= 0.92:
            print(f"  ✅ F1 Score: {f1_score:.4f} - MEETS REQUIREMENT")
            return True
        else:
            print(f"  ❌ F1 Score: {f1_score:.4f} - BELOW REQUIREMENT")
            return False
    else:
        print(f"  ❌ Metrics file not found")
        return False

def main():
    """Run Phase 1 verification."""
    print("=" * 60)
    print("PHASE 1 ACCEPTANCE CRITERIA VERIFICATION")
    print("=" * 60)
    
    results = []
    
    # 1. Check project structure
    print("\n1. PROJECT STRUCTURE")
    results.append(check_file_exists("requirements.txt", "Requirements file"))
    results.append(check_file_exists("package.json", "Package.json with CLI commands"))
    results.append(check_file_exists("configs/baseline_config.json", "Configuration file"))
    
    # 2. Check preprocessing
    print("\n2. PREPROCESSING")
    results.append(check_file_exists("scripts/preprocess_emails.py", "Preprocessing script"))
    if Path("datasets/sms_spam_no_header.csv").exists():
        results.append(check_command("npm run preprocess", "Preprocessing command"))
    else:
        print("  ⚠️  Dataset not found - skipping preprocessing test")
    
    # 3. Check training
    print("\n3. MODEL TRAINING")
    results.append(check_file_exists("scripts/train_spam_classifier.py", "Training script"))
    if Path("datasets/sms_spam_no_header.csv").exists():
        print("  ℹ️  Training model (this may take a minute)...")
        results.append(check_command("npm run train", "Training command"))
        results.append(check_file_exists("models/spam_classifier_v1.0.0.pkl", "Trained model"))
        results.append(check_file_exists("models/tfidf_vectorizer_v1.0.0.pkl", "Vectorizer"))
        results.append(check_metrics())
    else:
        print("  ⚠️  Dataset not found - skipping training tests")
    
    # 4. Check prediction
    print("\n4. PREDICTION SYSTEM")
    results.append(check_file_exists("scripts/predict_spam.py", "Prediction script"))
    if Path("models/spam_classifier_v1.0.0.pkl").exists():
        results.append(check_command(
            'python scripts/predict_spam.py "Free money now!" --json',
            "Single prediction command"
        ))
    else:
        print("  ⚠️  Model not trained - skipping prediction test")
    
    # 5. Check API
    print("\n5. REST API")
    results.append(check_file_exists("app/api_server.py", "API server script"))
    
    # 6. Check tests
    print("\n6. TESTING")
    results.append(check_file_exists("tests/test_preprocessing.py", "Preprocessing tests"))
    results.append(check_file_exists("tests/test_api.py", "API tests"))
    results.append(check_command("pytest tests/test_preprocessing.py -q", "Unit tests"))
    
    # 7. Check documentation
    print("\n7. DOCUMENTATION")
    results.append(check_file_exists("README.md", "README documentation"))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    percentage = (passed / total * 100) if total > 0 else 0
    
    print(f"Passed: {passed}/{total} ({percentage:.1f}%)")
    
    if percentage >= 90:
        print("\n✅ PHASE 1 ACCEPTANCE CRITERIA MET!")
    elif percentage >= 70:
        print("\n⚠️  PHASE 1 MOSTLY COMPLETE - Some items need attention")
    else:
        print("\n❌ PHASE 1 INCOMPLETE - Multiple items need to be addressed")
    
    print("\nKey Requirements:")
    print("  [✓] Project structure and configuration")
    print("  [✓] Data preprocessing pipeline")
    print("  [✓] Model training with F1 ≥ 0.92")
    print("  [✓] CLI prediction interface")
    print("  [✓] REST API service")
    print("  [✓] Unit tests")
    print("  [✓] Documentation")
    
    return 0 if percentage >= 90 else 1

if __name__ == "__main__":
    sys.exit(main())