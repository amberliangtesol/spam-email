# phase1-baseline-classifier

## Summary
Establish the foundational spam classification system with end-to-end pipeline, CLI tools, and REST API, achieving baseline F1 ≥ 0.92.

## ADDED Requirements

### Data Preprocessing Pipeline
The system MUST provide a standardized text preprocessing pipeline for spam classification.

#### Scenario: Preprocess training dataset
Given a raw CSV file at `datasets/sms_spam_no_header.csv`
When running `npm run preprocess` or `python scripts/preprocess_emails.py`
Then the system should:
- Load and validate the CSV structure (label, text columns)
- Apply text normalization (lowercase, whitespace trimming)
- Mask special patterns (URLs → URL_TOKEN, emails → EMAIL_TOKEN, phones → PHONE_TOKEN)
- Save processed data to `datasets/processed/train_data.csv`
- Log preprocessing statistics (rows processed, patterns found)

#### Scenario: Handle edge cases in preprocessing
Given text with various edge cases:
- Empty strings
- Non-ASCII characters
- HTML tags
- Very long messages (>5000 chars)
When preprocessing
Then the system should gracefully handle all cases without errors

### Model Training System
The system MUST train a TF-IDF + Logistic Regression classifier with configurable parameters.

#### Scenario: Train baseline model
Given preprocessed training data
When running `npm run train` or `python scripts/train_spam_classifier.py`
Then the system should:
- Load configuration from `configs/baseline_config.json`
- Split data 80/20 for train/test with stratification
- Train TF-IDF vectorizer with max_features=5000
- Train Logistic Regression with random_state=42
- Save model to `models/spam_classifier_v1.0.0.pkl`
- Save vectorizer to `models/tfidf_vectorizer_v1.0.0.pkl`
- Generate evaluation metrics with F1 ≥ 0.92

#### Scenario: Generate evaluation report
Given a trained model and test set
When evaluation completes
Then create `reports/evaluation_v1.0.0.json` containing:
- Accuracy, Precision, Recall, F1 scores
- Confusion matrix values
- Classification report by class
- ROC AUC score
- Training timestamp and parameters

### CLI Prediction Interface
The system MUST provide command-line tools for single and batch predictions.

#### Scenario: Classify single message
Given a trained model exists
When running `npm run classify -- "You won a free iPhone!"`
Then output:
```
Label: spam
Probability: 0.987
Confidence: high
```

#### Scenario: Batch classify from CSV
Given a CSV file with text column
When running `npm run classify:csv -- ./datasets/inbox.csv`
Then:
- Process all messages in the file
- Add prediction and probability columns
- Save results to `./datasets/inbox_predictions.csv`
- Display summary statistics

### REST API Service
The system MUST expose a REST API for real-time predictions.

#### Scenario: Start API server
When running `npm run serve` or `python app/api_server.py`
Then:
- Start server on port 8000
- Load model and vectorizer into memory
- Log "API ready" with model version
- Respond to health checks at GET /health

#### Scenario: Single prediction endpoint
Given API is running
When sending POST /predict with body:
```json
{"text": "Congratulations! You've been selected!"}
```
Then respond with:
```json
{
  "label": "spam",
  "probability": 0.945,
  "confidence": "high",
  "model_version": "1.0.0",
  "processing_time_ms": 15
}
```

#### Scenario: Handle invalid API input
Given API is running
When sending invalid requests:
- Empty text field
- Missing text field
- Text longer than 10000 chars
Then return 422 with appropriate error message

### Model Persistence and Versioning
The system MUST properly save and load models with version tracking.

#### Scenario: Save model with metadata
When training completes
Then save:
- Model binary to `models/spam_classifier_v{version}.pkl`
- Vectorizer to `models/tfidf_vectorizer_v{version}.pkl`
- Config snapshot to `models/config_v{version}.json`
- Metrics to `models/metrics_v{version}.json`

#### Scenario: Load specific model version
Given multiple model versions exist
When starting prediction service
Then:
- Load model specified in environment or config
- Validate model compatibility
- Log loaded version details

### Configuration Management
The system MUST use configuration files for all parameters.

#### Scenario: Baseline configuration
Given `configs/baseline_config.json`:
```json
{
  "model": {
    "type": "LogisticRegression",
    "params": {"C": 1.0, "random_state": 42}
  },
  "vectorizer": {
    "max_features": 5000,
    "ngram_range": [1, 1],
    "min_df": 2
  },
  "validation": {
    "test_size": 0.2,
    "random_state": 42
  }
}
```
When training
Then all parameters should be applied as specified

### Testing and Validation
The system MUST include comprehensive tests for critical components.

#### Scenario: Unit test preprocessing
Given test suite in `tests/test_preprocessing.py`
When running `pytest tests/test_preprocessing.py`
Then verify:
- Pattern masking works correctly
- Text normalization is consistent
- Edge cases handled properly

#### Scenario: Integration test API
Given test suite in `tests/test_api.py`
When running `pytest tests/test_api.py`
Then verify:
- All endpoints respond correctly
- Error handling works
- Response formats match specification

### Documentation
The system MUST provide clear documentation for setup and usage.

#### Scenario: README documentation
Given `README.md` in project root
Then document:
- Installation steps
- Dataset setup
- Training commands
- Evaluation process
- API usage examples
- CLI usage examples
- Configuration options

## Acceptance Criteria
- [ ] F1 score ≥ 0.92 on test set
- [ ] All CLI commands functional (`preprocess`, `train`, `eval`, `classify`)
- [ ] API responds correctly to valid and invalid requests
- [ ] Models and configs properly versioned and saved
- [ ] Core unit tests passing
- [ ] README with complete usage instructions

## Dependencies
- Python 3.8+
- scikit-learn >= 1.0
- pandas >= 1.3
- numpy >= 1.21
- FastAPI or Flask
- pytest for testing

## Related Specs
- Phase 2: Recall Improvement (builds on this baseline)
- Phase 3: Precision Recovery (requires Phase 2)
- Phase 4: Visualization Dashboard (uses all models)