#!/usr/bin/env python3
"""REST API server for spam classification."""
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from scripts.predict_spam import SpamPredictor
from scripts.utils.config import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Spam Classifier API",
    description="REST API for spam/ham email classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
predictor: Optional[SpamPredictor] = None
model_load_time: Optional[datetime] = None
default_threshold: float = float(os.getenv("SPAM_THRESHOLD", "0.5"))


# Request/Response models
class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    text: str = Field(..., min_length=1, max_length=10000, 
                     description="Text to classify")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0,
                                      description="Optional classification threshold")
    
    @validator('text')
    def validate_text(cls, v):
        """Validate text input."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    label: str = Field(..., description="Predicted label (spam/ham)")
    probability: float = Field(..., ge=0.0, le=1.0, 
                             description="Spam probability")
    confidence: str = Field(..., description="Confidence level (high/medium/low)")
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    threshold_used: Optional[float] = Field(None, description="Actual threshold used for classification")


class BatchPredictRequest(BaseModel):
    """Request model for batch prediction endpoint."""
    texts: list[str] = Field(..., min_items=1, max_items=1000,
                           description="List of texts to classify")
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate text list."""
        cleaned = []
        for text in v:
            if text and text.strip():
                cleaned.append(text.strip()[:10000])  # Limit each text to 10000 chars
        if not cleaned:
            raise ValueError("No valid texts provided")
        return cleaned


class BatchPredictResponse(BaseModel):
    """Response model for batch prediction endpoint."""
    predictions: list[Dict[str, Any]] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of predictions")
    processing_time_ms: int = Field(..., description="Total processing time")


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    uptime_seconds: int = Field(..., description="Service uptime in seconds")
    default_threshold: float = Field(..., description="Default classification threshold")


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""
    model_version: str
    model_metrics: Dict[str, Any]
    loaded_at: str


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor, model_load_time
    
    try:
        logger.info("Loading model and vectorizer...")
        config_loader = ConfigLoader()
        
        # Initialize predictor
        predictor = SpamPredictor()
        model_load_time = datetime.now()
        
        logger.info(f"Model loaded successfully. Version: {predictor.version}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server")


# API endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Spam Classifier API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    global predictor, model_load_time
    
    uptime = 0
    if model_load_time:
        uptime = int((datetime.now() - model_load_time).total_seconds())
    
    return HealthResponse(
        status="healthy" if predictor else "unhealthy",
        model_loaded=predictor is not None,
        model_version=predictor.version if predictor else None,
        uptime_seconds=uptime,
        default_threshold=default_threshold
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """Predict spam/ham for a single text.
    
    Args:
        request: Prediction request with text
        
    Returns:
        Prediction response with label and probability
        
    Raises:
        HTTPException: If prediction fails
    """
    global predictor
    
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        start_time = time.time()
        
        # Make prediction
        result = predictor.predict_single(request.text)
        
        # Apply custom threshold if provided, otherwise use default
        threshold = request.threshold if request.threshold is not None else default_threshold
        if threshold != 0.5:  # Only override if not default
            result['label'] = 'spam' if result['probability'] > threshold else 'ham'
            result['threshold_used'] = threshold
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return PredictResponse(
            label=result['label'],
            probability=result['probability'],
            confidence=result['confidence'],
            model_version=result['model_version'],
            processing_time_ms=processing_time,
            threshold_used=result.get('threshold_used')
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """Predict spam/ham for multiple texts.
    
    Args:
        request: Batch prediction request with text list
        
    Returns:
        Batch prediction response with all results
        
    Raises:
        HTTPException: If prediction fails
    """
    global predictor
    
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        start_time = time.time()
        
        # Make predictions
        results = predictor.predict_batch(request.texts)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return BatchPredictResponse(
            predictions=results,
            total=len(results),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed"
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics():
    """Get model metrics.
    
    Returns:
        Model performance metrics
        
    Raises:
        HTTPException: If metrics not available
    """
    global predictor, model_load_time
    
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Load metrics file
        config_loader = ConfigLoader()
        version = predictor.version
        metrics_path = Path(config_loader.get('paths.model_dir')) / f"metrics_v{version}.json"
        
        if not metrics_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Metrics not found"
            )
        
        import json
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
        
        return MetricsResponse(
            model_version=version,
            model_metrics=metrics_data.get('metrics', {}),
            loaded_at=model_load_time.isoformat() if model_load_time else ""
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load metrics"
        )


def main():
    """Run the API server."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()