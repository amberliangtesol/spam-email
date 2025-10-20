# build-spam-classifier

## Summary
Build a production-ready spam email classifier with CLI and REST API interfaces, using TF-IDF and Logistic Regression, developed through 4 iterative phases with OpenSpec-driven development methodology.

## Why
Email spam continues to be a significant problem, wasting time and resources while potentially exposing users to phishing and malware. This project addresses the need for a reliable, interpretable, and easily deployable spam classification system. By using a phased approach with clear metrics, we ensure the solution is both effective (high recall to catch spam) and precise (low false positives). The dual CLI/API interfaces make it accessible to both developers and end-users, while the Streamlit dashboard provides transparency into model decisions.

## What Changes
- **New capabilities**: Complete spam classification system from scratch
- **CLI tools**: Scripts for preprocessing, training, prediction, and analysis
- **API service**: REST endpoints for real-time classification
- **Machine learning pipeline**: TF-IDF vectorization + Logistic Regression
- **Configuration system**: JSON-based parameter management
- **Visualization dashboard**: Interactive Streamlit application
- **Model persistence**: Versioned model storage and loading
- **Evaluation framework**: Metrics calculation and reporting

## Status
- **State**: draft
- **Author**: @amber
- **Created**: 2025-01-20
- **Phase**: Planning

## Goals
1. Establish end-to-end spam/ham classification pipeline with reproducible results
2. Provide dual interfaces: CLI (single/batch) and REST API (POST /predict)
3. Achieve baseline F1 ≥ 0.92, then iteratively improve Recall ≥ 0.93 and Precision ≥ 0.90
4. Create interactive Streamlit dashboard for visualization and analysis
5. Ensure full reproducibility with fixed seeds, versioned configs, and documentation

## Non-Goals
- Active/online learning systems (future work)
- Vector databases or embeddings (keep it simple with TF-IDF)
- Full MLOps pipeline with CI/CD (beyond MVP scope)
- Frontend web application (except Streamlit demo)
- Multi-language support (English only for MVP)

## Impact
### Users Affected
- Data scientists needing quick spam classification
- Developers integrating spam detection via API
- Business users analyzing spam patterns via dashboard

### Systems Changed
- New CLI tools for preprocessing, training, and prediction
- New HTTP API service for real-time classification
- New Streamlit app for interactive analysis

## Design Approach
### Development Methodology
- OpenSpec-driven: All changes tracked via proposals and spec deltas
- 4-phase iterative development with clear acceptance criteria
- Test-driven with measurable performance metrics

### Technical Architecture
```
Input → Preprocessing → TF-IDF Vectorization → Logistic Regression → Prediction
                            ↓                        ↓
                      Persistence              Model Export
                            ↓                        ↓
                       JSON/PKL               models/*.pkl
```

### Key Components
1. **Data Pipeline**: Normalized preprocessing with pattern masking
2. **Model Training**: Configurable hyperparameters with validation
3. **Inference Engine**: Batch and single prediction support
4. **API Service**: FastAPI/Flask with input validation
5. **Visualization**: Streamlit dashboard with metrics and analysis

## Implementation Phases

### Phase 1: Baseline Classifier
- End-to-end pipeline implementation
- CLI for single and batch prediction
- REST API with POST /predict endpoint
- Target: F1 ≥ 0.92, Accuracy ~97-98%
- Deliverables: Trained model, evaluation reports, basic tests

### Phase 2: Recall Optimization
- Hyperparameter tuning for class imbalance
- Configurable parameters (class_weight, ngram_range, min_df)
- Target: Recall ≥ 0.93
- Deliverables: Updated configs, improved metrics report

### Phase 3: Precision Recovery
- Threshold optimization via sweep analysis
- C-value tuning for regularization
- Target: Precision ≥ 0.90 with Recall ≥ 0.93
- Deliverables: Threshold analysis, confusion matrices, PR/ROC curves

### Phase 4: Visualization & Deployment
- Interactive Streamlit dashboard
- Real-time prediction interface
- Cloud deployment configuration
- Target: One-click deployment to Streamlit Cloud
- Deliverables: Live dashboard, deployment scripts

## Acceptance Criteria
### Phase 1
- [ ] `npm run preprocess/train/eval` commands functional
- [ ] `npm run classify -- "text"` returns label and probability
- [ ] API endpoint responds with proper JSON format
- [ ] F1 score ≥ 0.92 on test set
- [ ] Unit tests for preprocessing and inference pass

### Phase 2
- [ ] Configurable parameters in configs/ directory
- [ ] Recall ≥ 0.93 in metrics.json
- [ ] Class imbalance handling implemented

### Phase 3
- [ ] Precision ≥ 0.90 with Recall ≥ 0.93
- [ ] Threshold sweep report generated
- [ ] Confusion matrix and curves visualized

### Phase 4
- [ ] Streamlit app runs locally with latest model
- [ ] Successfully deployed to cloud
- [ ] All visualizations functional

## Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Dataset distribution mismatch | High | Validate on multiple spam datasets |
| Class imbalance | Medium | Use class_weight and threshold tuning |
| Model size for deployment | Low | Keep models under 100MB |
| API latency | Medium | Implement caching and batch processing |

## Dependencies
- Python 3.8+ with scikit-learn, pandas, numpy
- FastAPI or Flask for API service
- Streamlit for dashboard
- SMS Spam Collection dataset (UCI)

## Validation Plan
1. Unit tests for all preprocessing functions
2. Integration tests for API endpoints
3. Performance benchmarks for latency
4. Cross-validation for model stability
5. A/B testing framework for future improvements

## Timeline
- Phase 1: 2 days (baseline implementation)
- Phase 2: 1 day (recall optimization)
- Phase 3: 1 day (precision tuning)
- Phase 4: 2 days (visualization and deployment)
- Total: ~6 days of development

## Open Questions
1. Should we support multiple model types in parallel (e.g., SVM, Naive Bayes)?
2. What's the maximum acceptable latency for API predictions?
3. Should batch predictions return confidence scores for all classes?
4. Do we need model versioning beyond semantic versioning?

## References
- [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- [scikit-learn TF-IDF Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-cloud)