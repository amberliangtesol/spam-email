# Design: Spam Email Classifier System

## Architecture Overview

### System Components
```
┌─────────────────────────────────────────────────────────┐
│                     User Interfaces                      │
├──────────────┬────────────────┬─────────────────────────┤
│   CLI Tool   │   REST API     │   Streamlit Dashboard   │
└──────┬───────┴────────┬───────┴──────────┬──────────────┘
       │                │                  │
       v                v                  v
┌─────────────────────────────────────────────────────────┐
│                  Inference Engine                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Input Valid │→ │ Preprocessor │→ │  Predictor   │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────┐
│                    Model Storage                         │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ Trained Model│  │  Vectorizer  │  │   Configs   │  │
│  │  (.pkl)      │  │   (.pkl)     │  │   (.json)   │  │
│  └──────────────┘  └──────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

#### Training Pipeline
```
Raw Data → Preprocessing → Feature Extraction → Model Training → Evaluation
   │             │                │                    │              │
   v             v                v                    v              v
CSV File    Normalized      TF-IDF Matrix      LogisticReg      Metrics
            Text Data        + Labels            Model           Report
```

#### Inference Pipeline
```
Input Text → Validation → Preprocessing → Vectorization → Prediction
    │            │              │               │              │
    v            v              v               v              v
"Check this" → Length/     Normalize →    TF-IDF →      Classify
               Encoding     + Mask         Transform      + Prob
```

## Component Specifications

### 1. Data Preprocessing Module
**Location**: `scripts/preprocess_emails.py`

**Responsibilities**:
- Text normalization (lowercase, whitespace)
- Pattern masking (URLs, emails, phone numbers)
- Special character handling
- Optional stopword removal

**Key Functions**:
```python
def normalize_text(text: str) -> str:
    """Normalize text with consistent formatting"""
    
def mask_patterns(text: str) -> str:
    """Replace patterns with tokens (URL_TOKEN, EMAIL_TOKEN)"""
    
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full preprocessing pipeline to dataset"""
```

### 2. Model Training Module
**Location**: `scripts/train_spam_classifier.py`

**Configuration Schema**:
```json
{
  "model": {
    "type": "LogisticRegression",
    "params": {
      "C": 1.0,
      "class_weight": "balanced",
      "random_state": 42,
      "max_iter": 1000
    }
  },
  "vectorizer": {
    "max_features": 5000,
    "ngram_range": [1, 2],
    "min_df": 2,
    "max_df": 0.95,
    "use_idf": true
  },
  "preprocessing": {
    "remove_stopwords": false,
    "mask_patterns": true,
    "normalize": true
  },
  "validation": {
    "test_size": 0.2,
    "stratify": true,
    "random_state": 42
  }
}
```

### 3. Prediction Service
**Location**: `scripts/predict_spam.py`

**Input Validation Rules**:
- Min length: 1 character
- Max length: 10,000 characters
- Encoding: UTF-8
- Reject binary content

**Output Format**:
```json
{
  "label": "spam|ham",
  "probability": 0.95,
  "confidence": "high|medium|low",
  "model_version": "1.0.0",
  "processing_time_ms": 12
}
```

### 4. REST API Service
**Location**: `app/api_server.py`

**Endpoints**:
```
POST /predict
  Body: {"text": "string"}
  Response: {prediction object}
  
GET /health
  Response: {"status": "healthy", "model_loaded": true}
  
GET /metrics
  Response: {current performance metrics}
  
POST /predict/batch
  Body: {"texts": ["string1", "string2"]}
  Response: {"predictions": [array]}
```

**Error Handling**:
- 400: Invalid input format
- 422: Input validation failed
- 500: Model prediction error
- 503: Model not loaded

### 5. Streamlit Dashboard
**Location**: `app/streamlit_app.py`

**Pages/Sections**:
1. **Live Prediction**: Real-time classification interface
2. **Model Performance**: Metrics, confusion matrix, ROC/PR curves
3. **Data Analysis**: Class distribution, word frequencies
4. **Threshold Tuning**: Interactive threshold adjustment
5. **Batch Processing**: CSV upload and download

## Performance Specifications

### Latency Requirements
- Single prediction: < 50ms (p95)
- Batch (100 items): < 500ms
- API startup: < 5 seconds
- Model loading: < 2 seconds

### Resource Constraints
- Memory: < 512MB for API service
- Model size: < 100MB on disk
- CPU: Single core sufficient
- No GPU required

### Scalability Considerations
- Stateless API for horizontal scaling
- Model caching to reduce I/O
- Batch processing optimization
- Connection pooling for concurrent requests

## Security & Validation

### Input Sanitization
```python
def sanitize_input(text: str) -> str:
    # Remove control characters
    # Limit length
    # Validate encoding
    # Escape special characters
```

### Rate Limiting
- Per IP: 100 requests/minute
- Per API key: 1000 requests/hour
- Batch size limit: 1000 items

### Logging & Monitoring
- All predictions logged with timestamp
- Error tracking with context
- Performance metrics collection
- Model drift detection preparation

## Configuration Management

### Environment Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Configuration
MODEL_PATH=models/spam_classifier_v1.pkl
VECTORIZER_PATH=models/tfidf_vectorizer_v1.pkl
CONFIG_PATH=configs/model_config.json

# Feature Flags
ENABLE_BATCH_API=true
ENABLE_METRICS_ENDPOINT=true
LOG_PREDICTIONS=false
```

### Model Versioning Strategy
```
models/
├── spam_classifier_v1.0.0.pkl
├── tfidf_vectorizer_v1.0.0.pkl
├── config_v1.0.0.json
└── metrics_v1.0.0.json
```

## Testing Strategy

### Unit Tests
- Preprocessing functions (edge cases)
- Model loading/saving
- Input validation
- API response formatting

### Integration Tests
- End-to-end prediction flow
- API endpoint responses
- Batch processing
- Error handling paths

### Performance Tests
- Load testing (1000 req/s)
- Memory leak detection
- Model inference benchmarks
- Startup time validation

### Validation Tests
- Cross-validation scores
- Holdout set performance
- Edge case predictions
- Threshold sensitivity

## Deployment Architecture

### Local Development
```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train model
python scripts/train_spam_classifier.py

# Start API
python app/api_server.py

# Launch dashboard
streamlit run app/streamlit_app.py
```

### Cloud Deployment (Streamlit Cloud)
```yaml
# .streamlit/config.toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
```

### Docker Container
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app/api_server.py"]
```

## Future Enhancements

### Phase 5+ Considerations
1. **Model Improvements**:
   - Ensemble methods (RF + LR + NB)
   - Deep learning (BERT fine-tuning)
   - Multi-language support

2. **Feature Engineering**:
   - Metadata features (sender, time, subject)
   - Behavioral patterns
   - Link analysis

3. **Operations**:
   - A/B testing framework
   - Model registry
   - Automated retraining
   - Drift monitoring

4. **Integration**:
   - Email client plugins
   - Webhook notifications
   - Database persistence
   - Message queue integration