"""Model loader for Streamlit Cloud deployment."""
import os
import joblib
import pickle
from pathlib import Path
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

@st.cache_resource
def load_or_create_model():
    """Load existing model or create a demo model for Streamlit Cloud."""
    
    model_dir = Path('models')
    
    # Try to load existing model
    if model_dir.exists():
        model_files = list(model_dir.glob('*.pkl'))
        if model_files:
            try:
                # Sort by modification time and get the latest
                model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                model = joblib.load(model_files[0])
                st.success(f"✅ Loaded model: {model_files[0].name}")
                return model, None, model_files[0].name
            except Exception as e:
                st.warning(f"Could not load model file: {e}")
    
    # Train model using actual dataset
    st.info("🔨 Training model using SMS Spam dataset...")
    
    # Check if dataset exists
    dataset_path = Path('datasets/sms_spam_no_header.csv')
    if not dataset_path.exists():
        st.error("Dataset not found. Using demo model instead.")
        return create_demo_model()
    
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path, header=None, names=['label', 'message'])
        
        # Convert labels to binary (ham=0, spam=1)
        df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df['message'], 
            df['label_num'], 
            test_size=0.2, 
            random_state=42,
            stratify=df['label_num']
        )
        
        # Create pipeline with optimized parameters
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.90,
            use_idf=True,
            sublinear_tf=True,
            lowercase=True,
            strip_accents='unicode',
            token_pattern=r'\b\w+\b',
            norm='l2'
        )
        
        classifier = LogisticRegression(
            C=2.0,
            class_weight='balanced',
            random_state=42,
            solver='liblinear',
            penalty='l2',
            max_iter=3000,
            tol=0.0001
        )
        
        model = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
        
        # Save for future use
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / 'spam_classifier_v3.0.0.pkl'
        joblib.dump(model, model_path)
        
        return model, None, "spam_classifier_v3.0.0.pkl"
    
    except Exception as e:
        st.error(f"Error training model: {e}")
        st.info("Falling back to demo model...")
        return create_demo_model()

def create_demo_model():
    """Create a simple demo model as fallback."""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    classifier = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    
    model = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    # Train with sample data
    sample_texts = [
        # Spam examples
        "WIN FREE MONEY NOW! CLICK HERE!",
        "Congratulations! You've won $1000!",
        "URGENT: Claim your prize now!",
        "Get rich quick! Work from home!",
        "Free iPhone! Limited time offer!",
        # Ham examples
        "Hey, are we still meeting for lunch?",
        "Can you pick up milk on your way home?",
        "The meeting has been rescheduled to 3pm",
        "Happy birthday! Hope you have a great day!",
        "Thanks for your help with the project"
    ]
    
    sample_labels = [1, 1, 1, 1, 1,  # spam
                    0, 0, 0, 0, 0]   # ham
    
    model.fit(sample_texts, sample_labels)
    
    # Save for future use
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    demo_model_path = model_dir / 'demo_model.pkl'
    joblib.dump(model, demo_model_path)
    
    return model, None, "demo_model.pkl"

def get_sample_data():
    """Get sample data for demonstration."""
    return {
        'spam_examples': [
            "🎰 WIN BIG! Free casino chips waiting for you!",
            "💊 Cheap medications online! No prescription needed!",
            "💰 Make $5000 per week working from home!",
            "📱 Claim your free iPhone 15 Pro now!",
            "🏆 Congratulations! You've won the lottery!"
        ],
        'ham_examples': [
            "📅 Don't forget our meeting at 2pm today",
            "🛒 Can you grab some groceries on your way back?",
            "🎂 Happy birthday! Wishing you all the best!",
            "📧 Following up on our discussion yesterday",
            "☕ Want to grab coffee this weekend?"
        ]
    }