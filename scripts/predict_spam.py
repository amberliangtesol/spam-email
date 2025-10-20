#!/usr/bin/env python3
"""Prediction script for spam classification."""
import sys
import json
import joblib
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from scripts.utils.config import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpamPredictor:
    """Handles spam prediction for single and batch inputs."""
    
    def __init__(self, model_path: str = None, vectorizer_path: str = None, 
                 config_path: str = None):
        """Initialize predictor with model and vectorizer.
        
        Args:
            model_path: Path to trained model pickle file
            vectorizer_path: Path to trained vectorizer pickle file
            config_path: Path to configuration file
        """
        # Load configuration
        config_loader = ConfigLoader(config_path)
        self.config = config_loader.config
        
        # Set model paths
        if model_path is None:
            version = self.config.get('version', '1.0.0')
            model_dir = Path(self.config['paths']['model_dir'])
            model_path = model_dir / f"spam_classifier_v{version}.pkl"
        
        if vectorizer_path is None:
            version = self.config.get('version', '1.0.0')
            model_dir = Path(self.config['paths']['model_dir'])
            vectorizer_path = model_dir / f"tfidf_vectorizer_v{version}.pkl"
        
        # Load model and vectorizer
        self.model = self.load_model(model_path)
        self.vectorizer = self.load_vectorizer(vectorizer_path)
        self.version = self.config.get('version', '1.0.0')
        
        logger.info(f"Loaded model version {self.version}")
    
    def load_model(self, model_path: str) -> Any:
        """Load trained model from file.
        
        Args:
            model_path: Path to model pickle file
            
        Returns:
            Loaded model
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
    
    def load_vectorizer(self, vectorizer_path: str) -> Any:
        """Load trained vectorizer from file.
        
        Args:
            vectorizer_path: Path to vectorizer pickle file
            
        Returns:
            Loaded vectorizer
        """
        if not Path(vectorizer_path).exists():
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
        
        vectorizer = joblib.load(vectorizer_path)
        logger.info(f"Loaded vectorizer from {vectorizer_path}")
        return vectorizer
    
    def validate_input(self, text: str) -> str:
        """Validate and clean input text.
        
        Args:
            text: Input text to validate
            
        Returns:
            Validated text
            
        Raises:
            ValueError: If input is invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string")
        
        if len(text) > 10000:
            raise ValueError("Input text too long (max 10000 characters)")
        
        # Basic cleaning
        text = text.strip()
        
        if not text:
            raise ValueError("Input text is empty after cleaning")
        
        return text
    
    def get_confidence_level(self, probability: float) -> str:
        """Convert probability to confidence level.
        
        Args:
            probability: Prediction probability
            
        Returns:
            Confidence level (high/medium/low)
        """
        if probability >= 0.9 or probability <= 0.1:
            return "high"
        elif probability >= 0.7 or probability <= 0.3:
            return "medium"
        else:
            return "low"
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict spam/ham for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction results
        """
        # Validate input
        text = self.validate_input(text)
        
        # Transform text
        text_tfidf = self.vectorizer.transform([text])
        
        # Make prediction
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        
        # Get spam probability
        spam_prob = probability[1]
        
        # Prepare result
        result = {
            'label': 'spam' if prediction == 1 else 'ham',
            'probability': float(spam_prob),
            'confidence': self.get_confidence_level(spam_prob),
            'model_version': self.version
        }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict spam/ham for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.predict_single(text)
                result['index'] = i
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to predict text at index {i}: {e}")
                results.append({
                    'index': i,
                    'error': str(e),
                    'label': None,
                    'probability': None
                })
        
        return results
    
    def predict_csv(self, input_path: str, output_path: str = None,
                   text_column: str = 'text') -> pd.DataFrame:
        """Predict spam/ham for texts in CSV file.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save predictions (optional)
            text_column: Name of text column in CSV
            
        Returns:
            DataFrame with predictions
        """
        # Load CSV
        df = pd.read_csv(input_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")
        
        logger.info(f"Processing {len(df)} rows from {input_path}")
        
        # Make predictions
        predictions = []
        probabilities = []
        confidences = []
        
        for text in df[text_column]:
            try:
                result = self.predict_single(str(text))
                predictions.append(result['label'])
                probabilities.append(result['probability'])
                confidences.append(result['confidence'])
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                predictions.append(None)
                probabilities.append(None)
                confidences.append(None)
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        df['probability'] = probabilities
        df['confidence'] = confidences
        
        # Save if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved predictions to {output_path}")
        
        # Log summary statistics
        if predictions:
            spam_count = predictions.count('spam')
            ham_count = predictions.count('ham')
            logger.info(f"Predictions summary:")
            logger.info(f"  - Spam: {spam_count} ({spam_count/len(predictions)*100:.1f}%)")
            logger.info(f"  - Ham: {ham_count} ({ham_count/len(predictions)*100:.1f}%)")
        
        return df


def format_output(result: Dict[str, Any], verbose: bool = False) -> str:
    """Format prediction result for display.
    
    Args:
        result: Prediction result dictionary
        verbose: Whether to show detailed output
        
    Returns:
        Formatted string
    """
    if verbose:
        output = f"""
Prediction Result:
  Label: {result['label'].upper()}
  Probability: {result['probability']:.3f}
  Confidence: {result['confidence']}
  Model Version: {result['model_version']}
"""
    else:
        output = f"Label: {result['label']}\nProbability: {result['probability']:.3f}\nConfidence: {result['confidence']}"
    
    return output.strip()


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(description='Predict spam/ham for text messages')
    parser.add_argument('text', nargs='?', help='Text to classify')
    parser.add_argument('--csv', type=str, help='Path to CSV file for batch prediction')
    parser.add_argument('--output', type=str, help='Output path for CSV predictions')
    parser.add_argument('--text-column', type=str, default='text', 
                       help='Name of text column in CSV')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--vectorizer', type=str, help='Path to vectorizer file')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = SpamPredictor(
            model_path=args.model,
            vectorizer_path=args.vectorizer,
            config_path=args.config
        )
        
        if args.csv:
            # Batch prediction from CSV
            output_path = args.output or args.csv.replace('.csv', '_predictions.csv')
            df = predictor.predict_csv(args.csv, output_path, args.text_column)
            
            if not args.json:
                print(f"\nProcessed {len(df)} messages")
                print(f"Results saved to: {output_path}")
            else:
                # Output summary as JSON
                summary = {
                    'total': len(df),
                    'spam': int((df['prediction'] == 'spam').sum()),
                    'ham': int((df['prediction'] == 'ham').sum()),
                    'output_file': output_path
                }
                print(json.dumps(summary, indent=2))
        
        elif args.text:
            # Single text prediction
            result = predictor.predict_single(args.text)
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(format_output(result, args.verbose))
        
        else:
            # Read from stdin
            print("Enter text to classify (Ctrl+D to finish):")
            text = sys.stdin.read().strip()
            
            if text:
                result = predictor.predict_single(text)
                
                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print(format_output(result, args.verbose))
            else:
                print("No text provided")
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        if args.json:
            error_response = {'error': str(e)}
            print(json.dumps(error_response))
        return 1


if __name__ == "__main__":
    sys.exit(main())