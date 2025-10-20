#!/usr/bin/env python3
"""
Live Prediction Page - Real-time spam classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from pathlib import Path
import plotly.graph_objects as go
import re
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Live Prediction - Spam Classifier",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS for this page
st.markdown("""
<style>
    .prediction-box {
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .spam-box {
        background-color: #ffebee;
        border: 2px solid #ef5350;
    }
    .ham-box {
        background-color: #e8f5e9;
        border: 2px solid #66bb6a;
    }
    .confidence-high {
        color: #2e7d32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .confidence-low {
        color: #c62828;
        font-weight: bold;
    }
    .feature-highlight {
        background-color: #fff3cd;
        padding: 0.2rem 0.4rem;
        border-radius: 0.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and vectorizer"""
    try:
        # Import model_loader from parent directory
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from model_loader import load_or_create_model
        
        # Load or create model using the unified loader
        model, vectorizer, model_name = load_or_create_model()
        
        # Return model and vectorizer (None if pipeline)
        return model, vectorizer
        
    except ImportError as e:
        # Fallback to old method if model_loader not found
        model_dir = Path('models')
        
        # Try to find the latest model
        model_files = list(model_dir.glob('spam_classifier*.pkl'))
        if not model_files:
            model_files = list(model_dir.glob('*precision*.pkl'))
        if not model_files:
            model_files = list(model_dir.glob('*.pkl'))
            
        if not model_files:
            return None, None
        
        # Sort by modification time and get the latest
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        model_path = model_files[0]
        
        model = joblib.load(model_path)
        
        # Check if model is a pipeline with vectorizer
        if hasattr(model, 'named_steps') and 'vectorizer' in model.named_steps:
            return model, None
        
        # Otherwise, load separate vectorizer
        vectorizer_files = list(model_dir.glob('*vectorizer*.pkl'))
        if vectorizer_files:
            vectorizer_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            vectorizer = joblib.load(vectorizer_files[0])
            return model, vectorizer
        
        return model, None
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_spam(text, model, vectorizer=None, threshold=0.5):
    """Make prediction on text"""
    try:
        start_time = time.time()
        
        if vectorizer:
            # Separate model and vectorizer
            features = vectorizer.transform([text])
            proba = model.predict_proba(features)[0]
        else:
            # Pipeline model
            proba = model.predict_proba([text])[0]
        
        spam_probability = proba[1]
        label = 'spam' if spam_probability > threshold else 'ham'
        
        # Determine confidence level
        confidence_score = max(proba)
        if confidence_score > 0.9:
            confidence = 'high'
        elif confidence_score > 0.7:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        return {
            'label': label,
            'spam_probability': spam_probability,
            'ham_probability': proba[0],
            'confidence': confidence,
            'confidence_score': confidence_score,
            'processing_time': processing_time,
            'threshold': threshold
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def extract_top_features(text, model, vectorizer, n_features=10):
    """Extract top contributing features for the prediction"""
    try:
        if vectorizer is None and hasattr(model, 'named_steps'):
            vectorizer = model.named_steps.get('vectorizer')
            classifier = model.named_steps.get('classifier', model.named_steps.get('model'))
        else:
            classifier = model
        
        if not vectorizer or not hasattr(classifier, 'coef_'):
            return [], []
        
        # Transform text
        features = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get feature weights
        coef = classifier.coef_[0]
        
        # Get non-zero features for this text
        nonzero_indices = features.nonzero()[1]
        
        if len(nonzero_indices) == 0:
            return [], []
        
        # Calculate contribution of each feature
        contributions = []
        for idx in nonzero_indices:
            feature = feature_names[idx]
            weight = coef[idx] * features[0, idx]
            contributions.append((feature, weight))
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Separate spam and ham indicators
        spam_features = [(f, w) for f, w in contributions if w > 0][:n_features]
        ham_features = [(f, w) for f, w in contributions if w < 0][:n_features]
        
        return spam_features, ham_features
        
    except Exception as e:
        st.error(f"Feature extraction error: {str(e)}")
        return [], []

def highlight_features_in_text(text, features):
    """Highlight important features in the original text"""
    highlighted_text = text
    for feature, _ in features:
        # Case-insensitive replacement with highlighting
        pattern = re.compile(re.escape(feature), re.IGNORECASE)
        highlighted_text = pattern.sub(
            f'<span class="feature-highlight">{feature}</span>',
            highlighted_text
        )
    return highlighted_text

def main():
    st.title("🔮 Live Spam Prediction")
    st.markdown("Classify messages in real-time with detailed explanations")
    
    # Load model
    model, vectorizer = load_model_and_vectorizer()
    
    if model is None:
        st.error("❌ No trained model found. Please train a model first.")
        st.stop()
    
    # Sidebar configuration
    st.sidebar.markdown("### ⚙️ Prediction Settings")
    
    # Threshold slider
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Messages with spam probability above this threshold are classified as spam"
    )
    
    # Show explanation toggle
    show_explanation = st.sidebar.checkbox("Show Detailed Explanation", value=True)
    show_features = st.sidebar.checkbox("Highlight Important Words", value=True)
    
    # Prediction history in session state
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Main prediction interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Enter Message")
        
        # Text input
        text_input = st.text_area(
            "Message to classify:",
            height=150,
            placeholder="Enter or paste your message here...\n\nExample: 'Congratulations! You've won a free iPhone. Click here to claim your prize!'",
            key="text_input"
        )
        
        # Sample messages
        st.markdown("#### 🎲 Try Sample Messages")
        sample_col1, sample_col2 = st.columns(2)
        
        with sample_col1:
            if st.button("🎁 Spam Sample", use_container_width=True):
                st.session_state.text_input = "WINNER!! You have won £1000 cash or a £2000 prize. To claim, call 09050000555. Valid 12 hours only."
        
        with sample_col2:
            if st.button("✉️ Ham Sample", use_container_width=True):
                st.session_state.text_input = "Hi! How are you? Just wanted to check if we're still meeting for lunch tomorrow at noon."
        
        # Analyze button
        analyze_button = st.button(
            "🔍 Analyze Message",
            type="primary",
            use_container_width=True,
            disabled=not text_input
        )
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        if st.session_state.prediction_history:
            total_predictions = len(st.session_state.prediction_history)
            spam_count = sum(1 for p in st.session_state.prediction_history if p['label'] == 'spam')
            ham_count = total_predictions - spam_count
            
            st.metric("Total Analyzed", total_predictions)
            col_spam, col_ham = st.columns(2)
            with col_spam:
                st.metric("Spam", spam_count, delta=f"{spam_count/total_predictions*100:.1f}%")
            with col_ham:
                st.metric("Ham", ham_count, delta=f"{ham_count/total_predictions*100:.1f}%")
        else:
            st.info("No predictions yet")
    
    # Prediction results
    if analyze_button and text_input:
        with st.spinner("Analyzing message..."):
            result = predict_spam(text_input, model, vectorizer, threshold)
        
        if result:
            # Add to history
            st.session_state.prediction_history.append({
                'timestamp': datetime.now(),
                'text': text_input[:100] + '...' if len(text_input) > 100 else text_input,
                **result
            })
            
            # Display result
            st.markdown("---")
            st.markdown("## 🎯 Prediction Result")
            
            # Main result box
            if result['label'] == 'spam':
                st.markdown(
                    f'<div class="prediction-box spam-box">'
                    f'<h1>🚫 SPAM</h1>'
                    f'<p>Confidence: {result["confidence_score"]:.1%}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-box ham-box">'
                    f'<h1>✅ HAM (Legitimate)</h1>'
                    f'<p>Confidence: {result["confidence_score"]:.1%}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Probability gauge
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = result['spam_probability'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Spam Probability (%)"},
                    delta = {'reference': threshold * 100, 'increasing': {'color': "red"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 25], 'color': '#e8f5e9'},
                            {'range': [25, 50], 'color': '#fff3cd'},
                            {'range': [50, 75], 'color': '#ffe0b2'},
                            {'range': [75, 100], 'color': '#ffebee'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            # Details
            st.markdown("### 📈 Classification Details")
            detail_col1, detail_col2, detail_col3 = st.columns(3)
            
            with detail_col1:
                st.metric("Spam Probability", f"{result['spam_probability']:.3f}")
            with detail_col2:
                st.metric("Ham Probability", f"{result['ham_probability']:.3f}")
            with detail_col3:
                st.metric("Processing Time", f"{result['processing_time']:.1f} ms")
            
            # Feature explanation
            if show_explanation:
                st.markdown("### 🔍 Prediction Explanation")
                
                spam_features, ham_features = extract_top_features(
                    text_input, model, vectorizer, n_features=10
                )
                
                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    st.markdown("#### 🚫 Spam Indicators")
                    if spam_features:
                        for feature, weight in spam_features[:5]:
                            st.markdown(f"• **{feature}** (weight: {weight:.3f})")
                    else:
                        st.info("No strong spam indicators found")
                
                with exp_col2:
                    st.markdown("#### ✅ Ham Indicators")
                    if ham_features:
                        for feature, weight in ham_features[:5]:
                            st.markdown(f"• **{feature}** (weight: {abs(weight):.3f})")
                    else:
                        st.info("No strong ham indicators found")
                
                # Highlighted text
                if show_features and (spam_features or ham_features):
                    st.markdown("#### 📝 Message with Highlighted Features")
                    all_features = spam_features[:5] + ham_features[:5]
                    highlighted = highlight_features_in_text(text_input, all_features)
                    st.markdown(
                        f'<div style="padding: 1rem; background-color: #f5f5f5; '
                        f'border-radius: 0.5rem; line-height: 1.6;">{highlighted}</div>',
                        unsafe_allow_html=True
                    )
    
    # Prediction history
    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("### 📜 Prediction History")
        
        # Convert to dataframe
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        # Display table
        display_df = history_df[['timestamp', 'text', 'label', 'spam_probability', 'processing_time']].head(10)
        display_df['spam_probability'] = display_df['spam_probability'].apply(lambda x: f"{x:.1%}")
        display_df['processing_time'] = display_df['processing_time'].apply(lambda x: f"{x:.1f} ms")
        display_df.columns = ['Time', 'Message', 'Prediction', 'Spam %', 'Time (ms)']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

if __name__ == "__main__":
    main()