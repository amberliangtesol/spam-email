# phase3-precision-recovery

## Summary
Optimize the classifier to achieve Precision ≥ 0.90 while maintaining Recall ≥ 0.93 through threshold tuning, regularization adjustments, and comparative model analysis.

## MODIFIED Requirements

### Threshold Optimization System
The system MUST implement comprehensive threshold tuning to balance precision and recall.

#### Scenario: Threshold sweep analysis
When running `npm run sweep:threshold` or `python scripts/threshold_sweep.py`
Then:
- Test thresholds from 0.1 to 0.9 in 0.01 increments
- Calculate precision, recall, F1 at each threshold
- Find optimal threshold for Precision ≥ 0.90 with Recall ≥ 0.93
- Generate `reports/threshold_sweep.csv` with all results
- Save best threshold to `configs/optimal_threshold.json`

#### Scenario: Apply optimal threshold
Given optimal threshold of 0.45 (example)
When making predictions
Then:
- Use `predict_proba` and apply custom threshold
- Label as spam if probability > threshold
- Achieve target Precision ≥ 0.90, Recall ≥ 0.93

### Regularization Fine-tuning
The system MUST optimize regularization parameters for better precision.

#### Scenario: C-value optimization
Given C values to test: [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
When running `npm run tune:regularization`
Then:
- Train model with each C value
- Evaluate precision-recall trade-offs
- Select C that maximizes precision with recall constraint
- Document in `reports/regularization_analysis.json`

#### Scenario: L1 vs L2 regularization comparison
When comparing regularization types:
- L2 (ridge) with various C values
- L1 (lasso) for feature selection
- ElasticNet (L1+L2 combined)
Then generate comparison showing:
- Number of features selected
- Precision/Recall achievements
- Model interpretability differences

### Advanced Model Comparison
The system MUST evaluate alternative algorithms for precision improvement.

#### Scenario: LinearSVC comparison
When training LinearSVC model:
```python
from sklearn.svm import LinearSVC
model = LinearSVC(C=1.0, class_weight='balanced', max_iter=5000)
```
Then compare with Logistic Regression:
- Training time
- Prediction latency
- Precision/Recall scores
- Decision boundary characteristics

#### Scenario: Ensemble consideration
When evaluating ensemble approach:
- Voting classifier with LR + SVC
- Calibrated classifier for probability adjustment
Then document whether ensemble improves metrics

### Precision-focused Feature Engineering
The system MUST implement features specifically for reducing false positives.

#### Scenario: Spam confidence scoring
Given high-confidence spam indicators:
- Multiple exclamation marks (!!!)
- ALL CAPS percentage > 30%
- Known spam phrases ("act now", "limited time")
- Suspicious URL patterns
When engineering features
Then:
- Create binary flags for strong indicators
- Add to feature set alongside TF-IDF
- Measure impact on precision

#### Scenario: Length-based features
When adding message length features:
- Very short messages (< 10 words)
- Very long messages (> 200 words)
- Character count
- Average word length
Then evaluate if these improve precision

### Visualization of Performance Curves
The system MUST generate visual representations of model performance.

#### Scenario: Generate PR curve
When running `npm run visualize:curves`
Then create `reports/pr_curve.png` showing:
- Precision-Recall curve
- Area under PR curve (AUPRC)
- Optimal threshold point marked
- Baseline (random) performance line

#### Scenario: Generate ROC curve
When creating ROC visualization
Then include:
- True Positive Rate vs False Positive Rate
- AUC-ROC score
- Optimal threshold point
- Comparison with previous phases

#### Scenario: Confusion matrix heatmap
When generating confusion matrix
Then create `reports/confusion_matrix.png` with:
- Actual vs Predicted labels
- Counts and percentages
- Color coding for magnitude
- Clear labels for spam/ham

### Cost-sensitive Learning
The system MUST support configurable misclassification costs.

#### Scenario: Define cost matrix
Given business requirements:
- False Positive cost: 1.0 (legitimate email marked as spam)
- False Negative cost: 0.3 (spam reaches inbox)
When training with cost matrix
Then optimize for minimum total cost

#### Scenario: Report cost-based metrics
When evaluating model
Then report:
- Total misclassification cost
- Cost per false positive
- Cost per false negative
- Cost reduction vs baseline

### Production-ready Threshold Configuration
The system MUST make threshold configurable for production use.

#### Scenario: Runtime threshold adjustment
Given API running with model
When receiving request with optional threshold:
```json
{
  "text": "Check this message",
  "threshold": 0.45
}
```
Then apply specified threshold for classification

#### Scenario: Environment-based threshold
Given environment variable `SPAM_THRESHOLD=0.45`
When starting API service
Then use this threshold as default

## ADDED Requirements

### Performance Benchmarking Suite
The system MUST provide comprehensive performance benchmarks.

#### Scenario: Generate full benchmark report
When running `npm run benchmark`
Then create `reports/benchmark_results.md` containing:
- Metrics across all three phases
- Best configuration for each metric
- Trade-off analysis
- Recommendations for production

### Model Explainability
The system MUST provide explanations for predictions.

#### Scenario: Explain individual prediction
Given a message classified as spam
When requesting explanation
Then provide:
- Top contributing words/n-grams
- Probability breakdown
- Feature weights applied
- Confidence level justification

## Acceptance Criteria
- [ ] Precision ≥ 0.90 achieved
- [ ] Recall ≥ 0.93 maintained
- [ ] Optimal threshold determined and documented
- [ ] PR and ROC curves generated
- [ ] Confusion matrix visualization created
- [ ] Threshold made configurable in API
- [ ] Comparison with LinearSVC completed
- [ ] All Phase 1 & 2 functionality preserved

## Dependencies
- All Phase 1 & 2 dependencies
- matplotlib/seaborn for visualization
- scikit-learn metrics module

## Related Specs
- Phase 1: Baseline Classifier (foundation)
- Phase 2: Recall Improvement (prerequisite)
- Phase 4: Interactive Dashboard (uses optimized models)