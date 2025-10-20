"""Unit tests for preprocessing module."""
import sys
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.preprocess_emails import TextPreprocessor
import pandas as pd


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""
    
    def setup_method(self):
        """Setup test preprocessor."""
        self.preprocessor = TextPreprocessor()
    
    def test_normalize_text(self):
        """Test text normalization."""
        # Test lowercase conversion
        assert self.preprocessor.normalize_text("HELLO WORLD") == "hello world"
        
        # Test whitespace normalization
        assert self.preprocessor.normalize_text("  hello   world  ") == "hello world"
        
        # Test multiple spaces
        assert self.preprocessor.normalize_text("hello\t\nworld") == "hello world"
        
        # Test empty string
        assert self.preprocessor.normalize_text("") == ""
        
        # Test None handling
        assert self.preprocessor.normalize_text(None) == ""
    
    def test_mask_patterns(self):
        """Test pattern masking."""
        # Test URL masking
        text = "Visit https://example.com for more"
        result = self.preprocessor.mask_patterns(text)
        assert "URL_TOKEN" in result
        assert "https://example.com" not in result
        
        # Test email masking
        text = "Contact us at test@example.com"
        result = self.preprocessor.mask_patterns(text)
        assert "EMAIL_TOKEN" in result
        assert "test@example.com" not in result
        
        # Test phone masking
        text = "Call us at +1-555-123-4567"
        result = self.preprocessor.mask_patterns(text)
        assert "PHONE_TOKEN" in result
        
        # Test number masking
        text = "You won 1000 dollars"
        result = self.preprocessor.mask_patterns(text)
        assert "NUMBER_TOKEN" in result
        assert "1000" not in result
    
    def test_preprocess_text(self):
        """Test full preprocessing pipeline."""
        text = "WIN $1000 NOW! Visit http://scam.com or email win@prize.com"
        result = self.preprocessor.preprocess_text(text)
        
        # Check normalization
        assert result.islower()
        
        # Check pattern masking
        assert "URL_TOKEN" in result
        assert "EMAIL_TOKEN" in result
        assert "NUMBER_TOKEN" in result
        assert "http://scam.com" not in result
        assert "win@prize.com" not in result
    
    def test_preprocess_dataset(self):
        """Test dataset preprocessing."""
        # Create test dataframe
        df = pd.DataFrame({
            'label': ['spam', 'ham', 'spam'],
            'text': [
                "FREE MONEY at http://win.com",
                "Hello, how are you?",
                "Call 555-1234 for prizes!"
            ]
        })
        
        # Preprocess dataset
        result = self.preprocessor.preprocess_dataset(df)
        
        # Check that all texts are processed
        assert len(result) == 3
        assert "URL_TOKEN" in result.iloc[0]['text']
        assert "PHONE_TOKEN" in result.iloc[2]['text']
        
        # Check statistics
        assert self.preprocessor.stats['total_processed'] == 3
        assert self.preprocessor.stats['urls_found'] > 0
        assert self.preprocessor.stats['phones_found'] > 0
    
    def test_edge_cases(self):
        """Test edge cases in preprocessing."""
        # Very long text
        long_text = "spam " * 5000
        result = self.preprocessor.preprocess_text(long_text)
        assert len(result) > 0
        
        # Special characters
        special_text = "Spëcîál çháracters & symbols! @#$%"
        result = self.preprocessor.preprocess_text(special_text)
        assert len(result) > 0
        
        # HTML tags
        html_text = "<p>This is <b>HTML</b> content</p>"
        result = self.preprocessor.preprocess_text(html_text)
        assert "<" in result  # HTML not stripped in basic preprocessor
        
        # Empty and None
        assert self.preprocessor.preprocess_text("") == ""
        assert self.preprocessor.preprocess_text(None) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])