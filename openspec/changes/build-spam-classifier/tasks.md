# Implementation Tasks

## Phase 1: Baseline Classifier Foundation

### Setup & Configuration
- [x] Initialize Python project structure with directories (scripts/, models/, configs/, tests/, reports/, app/)
- [x] Create requirements.txt with core dependencies (scikit-learn, pandas, numpy, fastapi/flask)
- [x] Set up virtual environment and document setup instructions
- [x] Create baseline configuration file (configs/baseline_config.json)
- [x] Implement configuration loader utility (scripts/utils/config.py)

### Data Pipeline
- [x] Create data validation script to verify dataset format
- [x] Implement preprocessing module (scripts/preprocess_emails.py)
  - [x] Text normalization (lowercase, whitespace)
  - [x] Pattern masking (URLs, emails, phones, numbers)
  - [x] Save processed data functionality
- [x] Write unit tests for preprocessing functions
- [x] Create preprocessing CLI command (npm run preprocess)

### Model Training
- [x] Implement training script (scripts/train_spam_classifier.py)
  - [x] Data loading and splitting (80/20)
  - [x] TF-IDF vectorizer training
  - [x] Logistic Regression training
  - [x] Model serialization (pickle)
- [x] Add model versioning system
- [x] Generate evaluation metrics and reports
- [x] Create training CLI command (npm run train)
- [x] Write unit tests for training pipeline

### Prediction System
- [x] Implement prediction script (scripts/predict_spam.py)
  - [x] Single text prediction
  - [x] Batch CSV prediction
  - [x] Confidence scoring
- [x] Create CLI commands (npm run classify, npm run classify:csv)
- [x] Add input validation and error handling
- [x] Write integration tests for prediction flow

### REST API
- [x] Set up FastAPI/Flask application structure
- [x] Implement /predict endpoint
  - [x] Request validation
  - [x] Model loading and caching
  - [x] Response formatting
- [x] Add /health endpoint
- [x] Implement error handling middleware
- [x] Create API startup script (npm run serve)
- [x] Write API integration tests

### Documentation & Testing
- [x] Create comprehensive README.md
- [x] Document all CLI commands and usage
- [x] Set up pytest configuration
- [x] Achieve F1 ≥ 0.92 baseline
- [x] Verify all acceptance criteria

## Phase 2: Recall Optimization

### Enhanced Configuration
- [ ] Extend configuration schema for advanced parameters
- [ ] Add support for class_weight configurations
- [ ] Implement ngram_range settings (bigrams, trigrams)
- [ ] Create recall-optimized config template
- [ ] Add configuration validation

### Hyperparameter Tuning
- [ ] Implement grid search script (scripts/tune_for_recall.py)
  - [ ] Define parameter grid
  - [ ] Cross-validation setup
  - [ ] Recall optimization
- [ ] Create automated tuning pipeline
- [ ] Save best parameters to config
- [ ] Generate tuning report
- [ ] Create tuning CLI command (npm run tune:recall)

### Feature Engineering
- [ ] Implement n-gram extraction (bigrams minimum)
- [ ] Add feature importance analysis
- [ ] Create feature analysis script
- [ ] Generate top spam indicators report
- [ ] Implement error analysis for false negatives

### Model Improvements
- [ ] Apply class balancing techniques
- [ ] Experiment with different solvers
- [ ] Implement model comparison framework
- [ ] Track metrics improvements
- [ ] Achieve Recall ≥ 0.93

### Analysis Tools
- [ ] Create misclassification analyzer
- [ ] Generate feature importance reports
- [ ] Implement threshold exploration tool
- [ ] Add model comparison utilities
- [ ] Document recall improvements

## Phase 3: Precision Recovery

### Threshold Optimization
- [ ] Implement threshold sweep script (scripts/threshold_sweep.py)
  - [ ] Test range 0.1 to 0.9 in 0.01 increments
  - [ ] Calculate metrics at each threshold
  - [ ] Find optimal balance point
- [ ] Save optimal threshold to config
- [ ] Apply threshold in prediction pipeline
- [ ] Create threshold CLI command (npm run sweep:threshold)

### Regularization Tuning
- [ ] Test multiple C values for regularization
- [ ] Compare L1 vs L2 regularization
- [ ] Implement regularization analysis script
- [ ] Document impact on precision
- [ ] Select optimal regularization parameters

### Model Comparison
- [ ] Implement LinearSVC training
- [ ] Compare with Logistic Regression
- [ ] Test ensemble approaches (optional)
- [ ] Generate comparison reports
- [ ] Select best model for production

### Visualization Generation
- [ ] Create PR curve visualization
- [ ] Generate ROC curve with AUC
- [ ] Implement confusion matrix heatmap
- [ ] Add threshold visualization on curves
- [ ] Save all plots to reports/

### Advanced Features
- [ ] Add cost-sensitive learning support
- [ ] Implement configurable thresholds in API
- [ ] Add prediction explanation capability
- [ ] Create benchmark suite
- [ ] Achieve Precision ≥ 0.90, Recall ≥ 0.93

## Phase 4: Interactive Dashboard

### Streamlit Setup
- [ ] Initialize Streamlit application structure
- [ ] Create multi-page navigation
- [ ] Set up theme configuration (.streamlit/config.toml)
- [ ] Implement model loading and caching
- [ ] Add session state management

### Live Prediction Page
- [ ] Create text input interface
- [ ] Implement real-time prediction
- [ ] Add confidence visualization
- [ ] Show prediction explanation
- [ ] Maintain prediction history

### Batch Processing Page
- [ ] Implement CSV file uploader
- [ ] Add file validation
- [ ] Create progress tracking
- [ ] Enable results download
- [ ] Display summary statistics

### Model Performance Page
- [ ] Display key metrics cards
- [ ] Create confusion matrix visualization
- [ ] Add classification report table
- [ ] Show model training history
- [ ] Implement metric comparisons

### Visualization Pages
- [ ] Create Performance Curves page (PR, ROC)
- [ ] Implement Feature Analysis page
- [ ] Add word cloud generations
- [ ] Create data distribution charts
- [ ] Enable plot interactivity with Plotly

### Threshold Tuning Page
- [ ] Implement interactive threshold slider
- [ ] Show real-time metric updates
- [ ] Display recommendations
- [ ] Add threshold comparison view
- [ ] Save selected threshold

### Data Explorer Page
- [ ] Show dataset statistics
- [ ] Implement message search
- [ ] Add filtering capabilities
- [ ] Create data export functions
- [ ] Display sample messages

### Deployment Preparation
- [ ] Prepare requirements.txt for Streamlit Cloud
- [ ] Configure environment variables
- [ ] Test local deployment
- [ ] Set up GitHub repository
- [ ] Deploy to Streamlit Cloud

### User Experience
- [ ] Add loading states and spinners
- [ ] Implement comprehensive error handling
- [ ] Ensure mobile responsiveness
- [ ] Add help documentation
- [ ] Create user tooltips

## Validation & Documentation

### Testing
- [ ] Run all unit tests
- [ ] Perform integration testing
- [ ] Validate all acceptance criteria
- [ ] Test on different datasets
- [ ] Verify API performance

### Documentation
- [ ] Update README with full instructions
- [ ] Document API endpoints
- [ ] Create user guide for dashboard
- [ ] Add configuration documentation
- [ ] Include troubleshooting guide

### Final Verification
- [ ] Verify Phase 1: F1 ≥ 0.92
- [ ] Verify Phase 2: Recall ≥ 0.93
- [ ] Verify Phase 3: Precision ≥ 0.90, Recall ≥ 0.93
- [ ] Verify Phase 4: Dashboard fully functional
- [ ] Ensure all npm commands work
- [ ] Confirm cloud deployment successful

## Parallel Work Opportunities

### Can be done in parallel:
- Documentation while coding
- Unit tests alongside implementation
- Visualization scripts independent of model
- API development parallel to CLI tools
- Streamlit pages can be developed independently

### Must be sequential:
- Phase 2 requires Phase 1 model
- Phase 3 requires Phase 2 improvements
- Phase 4 needs all models for comparison
- Deployment requires completed application

## Risk Mitigation Tasks

### Contingency Plans
- [ ] Create fallback for cloud deployment issues
- [ ] Implement model rollback mechanism
- [ ] Add data validation checkpoints
- [ ] Create backup configuration sets
- [ ] Document recovery procedures

### Performance Monitoring
- [ ] Add logging throughout pipeline
- [ ] Implement metrics tracking
- [ ] Create performance benchmarks
- [ ] Set up alert thresholds
- [ ] Document optimization opportunities