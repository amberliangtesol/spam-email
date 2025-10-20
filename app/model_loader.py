"""Model loader for Streamlit Cloud deployment."""
import os
import joblib
import pickle
from pathlib import Path
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

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
    
    # Create a demo model for deployment
    st.info("🔨 Creating demo model for Streamlit Cloud...")
    
    # Create a simple pipeline
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
        "You are our lucky winner!",
        "Make money fast! No experience needed!",
        "Discount pills online! Buy now!",
        "Hot singles in your area!",
        "Lose weight fast with this trick!",
        # Ham examples
        "Hey, are we still meeting for lunch?",
        "Can you pick up milk on your way home?",
        "The meeting has been rescheduled to 3pm",
        "Happy birthday! Hope you have a great day!",
        "Thanks for your help with the project",
        "See you at the conference tomorrow",
        "Your package has been delivered",
        "Reminder: Doctor appointment at 2pm",
        "Let me know if you need anything",
        "Great job on the presentation today!"
    ]
    
    sample_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # spam
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # ham
    
    # Fit the model
    model.fit(sample_texts, sample_labels)
    
    st.success("✅ Demo model created successfully!")
    
    # Save for future use
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