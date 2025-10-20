#!/usr/bin/env python3
"""Preprocessing module for spam email classification."""
import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Optional, Tuple
from utils.config import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Handles text preprocessing for spam classification."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration dictionary
        """
        if config is None:
            config_loader = ConfigLoader()
            config = config_loader.get('preprocessing', {})
        
        self.mask_patterns = config.get('mask_patterns', True)
        self.normalize = config.get('normalize', True)
        self.min_word_length = config.get('min_word_length', 1)
        
        # Pattern definitions
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}')
        self.number_pattern = re.compile(r'\b\d+\b')
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'urls_found': 0,
            'emails_found': 0,
            'phones_found': 0,
            'empty_texts': 0
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text with consistent formatting.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if pd.isna(text) or text == '':
            self.stats['empty_texts'] += 1
            return ''
        
        # Convert to string if not already
        text = str(text)
        
        # Lowercase
        text = text.lower()
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def mask_patterns(self, text: str) -> str:
        """Replace patterns with tokens.
        
        Args:
            text: Input text
            
        Returns:
            Text with masked patterns
        """
        # Count patterns for statistics
        self.stats['urls_found'] += len(self.url_pattern.findall(text))
        self.stats['emails_found'] += len(self.email_pattern.findall(text))
        self.stats['phones_found'] += len(self.phone_pattern.findall(text))
        
        # Replace patterns
        text = self.url_pattern.sub('URL_TOKEN', text)
        text = self.email_pattern.sub('EMAIL_TOKEN', text)
        text = self.phone_pattern.sub('PHONE_TOKEN', text)
        text = self.number_pattern.sub('NUMBER_TOKEN', text)
        
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Apply full preprocessing pipeline to text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        self.stats['total_processed'] += 1
        
        # Normalize
        if self.normalize:
            text = self.normalize_text(text)
        
        # Mask patterns
        if self.mask_patterns and text:
            text = self.mask_patterns(text)
        
        return text
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to entire dataset.
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            DataFrame with preprocessed text
        """
        logger.info(f"Starting preprocessing of {len(df)} messages")
        
        # Reset statistics
        self.stats = {key: 0 for key in self.stats}
        
        # Apply preprocessing
        df['text'] = df['text'].apply(self.preprocess_text)
        
        # Log statistics
        logger.info(f"Preprocessing complete. Statistics:")
        logger.info(f"  - Total messages: {self.stats['total_processed']}")
        logger.info(f"  - URLs found: {self.stats['urls_found']}")
        logger.info(f"  - Emails found: {self.stats['emails_found']}")
        logger.info(f"  - Phone numbers found: {self.stats['phones_found']}")
        logger.info(f"  - Empty texts: {self.stats['empty_texts']}")
        
        return df


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """Load and validate the spam dataset.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Validated DataFrame
    """
    try:
        # Load data
        df = pd.read_csv(file_path, header=None, names=['label', 'text'])
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Validate structure
        if 'label' not in df.columns or 'text' not in df.columns:
            raise ValueError("CSV must have 'label' and 'text' columns")
        
        # Validate labels
        unique_labels = df['label'].unique()
        if not set(unique_labels).issubset({'spam', 'ham'}):
            logger.warning(f"Unexpected labels found: {unique_labels}")
        
        # Check for missing values
        missing_texts = df['text'].isna().sum()
        missing_labels = df['label'].isna().sum()
        if missing_texts > 0:
            logger.warning(f"Found {missing_texts} missing text values")
        if missing_labels > 0:
            logger.warning(f"Found {missing_labels} missing label values")
        
        # Remove rows with missing values
        original_len = len(df)
        df = df.dropna()
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} rows with missing values")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Preprocess spam email dataset')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    parser.add_argument('--config', type=str, help='Configuration file path')
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    
    # Get file paths
    input_path = args.input or config_loader.get('paths.raw_data')
    output_path = args.output or config_loader.get('paths.processed_data')
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load and validate data
        df = load_and_validate_data(input_path)
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor(config_loader.get('preprocessing'))
        
        # Preprocess dataset
        df = preprocessor.preprocess_dataset(df)
        
        # Save processed data
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        # Display sample
        logger.info("Sample of processed data:")
        print(df.head())
        
        # Display class distribution
        class_dist = df['label'].value_counts()
        logger.info(f"Class distribution:")
        logger.info(f"  - Spam: {class_dist.get('spam', 0)}")
        logger.info(f"  - Ham: {class_dist.get('ham', 0)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())