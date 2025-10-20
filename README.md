# Spam Email Classifier

A production-ready spam email classifier with CLI and REST API interfaces, using TF-IDF and Logistic Regression for high-accuracy spam detection.

## Features

- **Dual Interfaces**: Command-line tools and REST API
- **High Accuracy**: Achieves F1 score ≥ 0.92 on test data
- **Fast Inference**: < 50ms for single predictions
- **Batch Processing**: Process CSV files with thousands of messages
- **Configurable**: JSON-based configuration for all parameters
- **Well-Tested**: Comprehensive unit and integration tests

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd spam-email
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
npm run setup
# or
pip install -r requirements.txt
```

### Dataset

The project uses the SMS Spam Collection dataset. Place it in `datasets/sms_spam_no_header.csv` with format:
```
label,text
ham,"Normal message text"
spam,"Spam message text"
```

### Training the Model

1. Preprocess the data:
```bash
npm run preprocess
```

2. Train the classifier:
```bash
npm run train
```

The trained model will be saved to `models/` with evaluation metrics in `reports/`.

## Usage

### CLI Commands

#### Single Text Classification
```bash
npm run classify -- "Check if this message is spam"
# or
python scripts/predict_spam.py "Your message here"
```

#### Batch CSV Classification
```bash
npm run classify:csv -- datasets/messages.csv
# or
python scripts/predict_spam.py --csv datasets/messages.csv --output predictions.csv
```

#### With JSON Output
```bash
python scripts/predict_spam.py "Test message" --json
```

### REST API

1. Start the API server:
```bash
npm run serve
# or
python app/api_server.py
```

2. Make predictions:
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "You won a free prize!"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Message 1", "Message 2", "Message 3"]}'

# Health check
curl "http://localhost:8000/health"
```

3. View API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Project Structure

```
spam-email/
├── datasets/           # Training and test data
├── models/            # Trained models and vectorizers
├── configs/           # Configuration files
├── scripts/           # CLI tools
│   ├── preprocess_emails.py
│   ├── train_spam_classifier.py
│   ├── predict_spam.py
│   └── utils/
├── app/               # API application
│   └── api_server.py
├── tests/             # Unit and integration tests
├── reports/           # Evaluation reports
└── openspec/          # Development specifications
```

## Configuration

Edit `configs/baseline_config.json` to customize:

```json
{
  "model": {
    "params": {
      "C": 1.0,              # Regularization strength
      "class_weight": null,   # Balance classes
      "random_state": 42
    }
  },
  "vectorizer": {
    "max_features": 5000,    # Vocabulary size
    "ngram_range": [1, 1],   # Unigrams only
    "min_df": 2              # Min document frequency
  }
}
```

## Model Performance

### Baseline Metrics (Phase 1)
- **Accuracy**: ~97-98%
- **F1 Score**: ≥ 0.92
- **Precision**: ~0.95
- **Recall**: ~0.90

### Performance Specifications
- Single prediction latency: < 50ms (p95)
- Batch processing: < 500ms for 100 items
- Model size: < 100MB
- Memory usage: < 512MB

## Testing

Run all tests:
```bash
npm run test
```

Run specific test suites:
```bash
# Preprocessing tests
pytest tests/test_preprocessing.py -v

# API tests
pytest tests/test_api.py -v
```

## Development Phases

This project follows a 4-phase development plan:

1. **Phase 1** ✅: Baseline classifier (F1 ≥ 0.92)
2. **Phase 2**: Recall optimization (Recall ≥ 0.93)
3. **Phase 3**: Precision tuning (Precision ≥ 0.90)
4. **Phase 4**: Interactive dashboard (Streamlit)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Single text prediction |
| `/predict/batch` | POST | Batch prediction |
| `/metrics` | GET | Model performance metrics |
| `/docs` | GET | Swagger documentation |

### Request/Response Examples

#### Prediction Request
```json
{
  "text": "Congratulations! You've won!",
  "threshold": 0.5  // optional
}
```

#### Prediction Response
```json
{
  "label": "spam",
  "probability": 0.987,
  "confidence": "high",
  "model_version": "1.0.0",
  "processing_time_ms": 15
}
```

## Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false

# Model Configuration (optional)
MODEL_PATH=models/spam_classifier_v1.0.0.pkl
VECTORIZER_PATH=models/tfidf_vectorizer_v1.0.0.pkl
```

## Troubleshooting

### Model Not Found
If you get "Model not found" errors:
1. Ensure you've run the training: `npm run train`
2. Check that model files exist in `models/` directory
3. Verify the version in `configs/baseline_config.json`

### Low F1 Score
If F1 score < 0.92:
1. Ensure data is properly preprocessed
2. Check class distribution in your dataset
3. Consider adjusting hyperparameters in config

### API Won't Start
1. Check port 8000 is not in use
2. Ensure all dependencies are installed
3. Verify Python version is 3.8+

## Contributing

See `openspec/changes/` for development specifications and guidelines.

## License

MIT

## Acknowledgments

- Dataset: [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) (UCI ML Repository)
- Built with scikit-learn, FastAPI, and pandas