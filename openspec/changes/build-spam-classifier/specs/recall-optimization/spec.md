# phase2-recall-improvement

## Summary
Enhance the spam classifier to achieve Recall ≥ 0.93 by implementing configurable hyperparameters and class imbalance handling techniques.

## MODIFIED Requirements

### Enhanced Configuration System
The configuration system MUST support advanced hyperparameters for improving recall.

#### Scenario: Extended configuration options
Given `configs/recall_optimized_config.json`:
```json
{
  "model": {
    "type": "LogisticRegression",
    "params": {
      "C": 0.5,
      "class_weight": "balanced",
      "random_state": 42,
      "solver": "liblinear",
      "max_iter": 2000
    }
  },
  "vectorizer": {
    "max_features": 8000,
    "ngram_range": [1, 2],
    "min_df": 1,
    "max_df": 0.95,
    "use_idf": true,
    "sublinear_tf": true
  },
  "preprocessing": {
    "remove_stopwords": false,
    "stem_words": false,
    "min_word_length": 2
  }
}
```
When training with this configuration
Then the model should apply all parameters correctly

#### Scenario: Class weight experimentation
Given the ability to set class_weight parameter
When trying values:
- "balanced" (automatic adjustment)
- {"spam": 2.0, "ham": 1.0} (manual weights)
- None (no weighting)
Then training should adjust sample weights accordingly

### Hyperparameter Tuning Module
The system MUST provide automated hyperparameter search for recall optimization.

#### Scenario: Grid search for optimal recall
Given a parameter grid:
```python
param_grid = {
    'C': [0.1, 0.5, 1.0, 2.0],
    'class_weight': ['balanced', {0:1, 1:2}, {0:1, 1:3}],
    'ngram_range': [(1,1), (1,2), (1,3)]
}
```
When running `npm run tune:recall` or `python scripts/tune_for_recall.py`
Then:
- Perform 5-fold cross-validation
- Optimize for recall score
- Save best parameters to `configs/best_recall_config.json`
- Generate tuning report with all combinations tried

#### Scenario: Track recall improvements
Given baseline model with Recall < 0.93
When applying optimized parameters
Then:
- New model achieves Recall ≥ 0.93
- Document trade-offs in precision
- Update `reports/metrics_comparison.json`

### N-gram Feature Expansion
The system MUST utilize n-gram features to capture more spam patterns.

#### Scenario: Bigram feature extraction
Given configuration with `ngram_range: [1, 2]`
When training the vectorizer
Then:
- Extract both unigrams and bigrams
- Apply min_df to filter rare n-grams
- Log top n-grams by class

#### Scenario: Trigram experimentation
Given configuration with `ngram_range: [1, 3]`
When comparing to bigram model
Then generate report showing:
- Model size differences
- Training time comparison
- Recall/Precision trade-offs
- Top trigram spam indicators

### Feature Analysis Tools
The system MUST provide tools to analyze which features contribute to recall.

#### Scenario: Generate feature importance report
When running `npm run analyze:features`
Then create `reports/feature_analysis.json` containing:
- Top 100 features for spam detection
- Features most often missed (false negatives)
- N-gram patterns unique to spam
- Word frequency distributions by class

#### Scenario: Misclassification analysis
Given false negative examples
When running `npm run analyze:errors`
Then:
- Identify common patterns in missed spam
- Generate recommendations for feature engineering
- Create `reports/error_analysis.md`

### Adaptive Threshold Exploration
The system MUST explore decision thresholds to optimize recall.

#### Scenario: Initial threshold analysis
Given trained model with default threshold (0.5)
When running `npm run analyze:threshold`
Then generate analysis showing:
- Recall at thresholds from 0.1 to 0.9
- Precision trade-offs at each threshold
- F1 scores across threshold range
- Save to `reports/threshold_analysis.csv`

### Validation with Imbalanced Metrics
The system MUST use appropriate metrics for imbalanced classification.

#### Scenario: Calculate comprehensive metrics
When evaluating model performance
Then report:
- Recall per class
- Precision per class
- F1 per class
- Macro/Micro/Weighted averages
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa score

#### Scenario: Cross-validation with stratification
When performing model validation
Then:
- Use StratifiedKFold with 5 splits
- Ensure class distribution preserved
- Report variance in recall across folds

## ADDED Requirements

### Incremental Model Comparison
The system MUST compare Phase 2 models against Phase 1 baseline.

#### Scenario: Generate comparison report
When running `npm run compare:models`
Then create `reports/model_comparison.html` showing:
- Side-by-side metrics (Phase 1 vs Phase 2)
- Confusion matrix comparison
- Processing time differences
- Model size comparison

### Early Warning System
The system MUST identify when recall drops below threshold.

#### Scenario: Monitor recall degradation
Given a minimum recall threshold of 0.93
When evaluating on new test data
Then:
- Alert if recall < 0.93
- Log specific failure patterns
- Suggest retraining if persistent

## Acceptance Criteria
- [ ] Recall ≥ 0.93 on test set
- [ ] Configurable hyperparameters working correctly
- [ ] N-gram features properly extracted (bigrams minimum)
- [ ] Class imbalance handling implemented
- [ ] Feature importance analysis available
- [ ] Model comparison report generated
- [ ] All Phase 1 functionality still working

## Dependencies
- All Phase 1 dependencies
- scikit-learn GridSearchCV
- imbalanced-learn (optional, for SMOTE exploration)

## Related Specs
- Phase 1: Baseline Classifier (prerequisite)
- Phase 3: Precision Recovery (builds on recall improvements)
- Phase 4: Visualization (displays all metrics)