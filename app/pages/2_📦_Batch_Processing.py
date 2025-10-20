#!/usr/bin/env python3
"""
Batch Processing Page - Process multiple messages from CSV
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Batch Processing - Spam Classifier",
    page_icon="📦",
    layout="wide"
)

@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and vectorizer"""
    try:
        # Import model_loader from parent directory
        from model_loader import load_or_create_model
        
        # Load or create model using the unified loader
        model, vectorizer, model_name = load_or_create_model()
        
        # Return model and vectorizer (None if pipeline)
        return model, vectorizer
        
    except ImportError:
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

def process_batch(df, text_column, model, vectorizer=None, threshold=0.5):
    """Process a batch of messages"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(df)
    chunk_size = 100  # Process in chunks for large files
    
    for i in range(0, total, chunk_size):
        chunk = df.iloc[i:min(i+chunk_size, total)]
        chunk_results = []
        
        for idx, row in chunk.iterrows():
            text = str(row[text_column])
            
            try:
                if vectorizer:
                    features = vectorizer.transform([text])
                    proba = model.predict_proba(features)[0]
                else:
                    proba = model.predict_proba([text])[0]
                
                spam_prob = proba[1]
                label = 'spam' if spam_prob > threshold else 'ham'
                confidence = max(proba)
                
                chunk_results.append({
                    'original_index': idx,
                    'text': text,
                    'prediction': label,
                    'spam_probability': spam_prob,
                    'ham_probability': proba[0],
                    'confidence': confidence
                })
                
            except Exception as e:
                chunk_results.append({
                    'original_index': idx,
                    'text': text,
                    'prediction': 'error',
                    'spam_probability': 0,
                    'ham_probability': 0,
                    'confidence': 0,
                    'error': str(e)
                })
        
        results.extend(chunk_results)
        
        # Update progress
        progress = min((i + len(chunk)) / total, 1.0)
        progress_bar.progress(progress)
        status_text.text(f'Processing: {min(i+chunk_size, total)}/{total} messages...')
    
    progress_bar.progress(1.0)
    status_text.text(f'✅ Completed: {total} messages processed')
    
    return pd.DataFrame(results)

def create_sample_csv():
    """Create a sample CSV for demonstration"""
    sample_data = {
        'message': [
            "Hey, are we still meeting for lunch tomorrow at 12?",
            "WINNER! You've won $1000. Click here to claim: bit.ly/win1000",
            "Your Amazon order has been shipped. Track your package.",
            "Urgent! Your account will be suspended. Verify now: phishing-site.com",
            "Can you pick up milk on your way home? Thanks!",
            "Congratulations! You're our lucky winner. Call 1-800-SCAM now!",
            "Meeting rescheduled to 3pm. See you in the conference room.",
            "Get rich quick! Make $5000 per week working from home!",
            "Happy birthday! Hope you have a wonderful day.",
            "Final notice: Your warranty is about to expire. Act now!"
        ],
        'sender': [
            "friend@email.com", "unknown@spam.com", "amazon@amazon.com",
            "fake@phishing.com", "spouse@email.com", "scammer@fake.com",
            "boss@company.com", "getrich@scam.com", "family@email.com",
            "warranty@spam.com"
        ]
    }
    return pd.DataFrame(sample_data)

def main():
    st.title("📦 Batch Message Processing")
    st.markdown("Process multiple messages at once from CSV files")
    
    # Load model
    model, vectorizer = load_model_and_vectorizer()
    
    if model is None:
        st.error("❌ No trained model found. Please train a model first.")
        st.stop()
    
    # Sidebar configuration
    st.sidebar.markdown("### ⚙️ Batch Processing Settings")
    
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )
    
    include_probabilities = st.sidebar.checkbox("Include Probability Scores", value=True)
    include_confidence = st.sidebar.checkbox("Include Confidence Scores", value=True)
    
    # Main interface
    tab1, tab2 = st.tabs(["📤 Upload & Process", "🎲 Use Sample Data"])
    
    with tab1:
        st.markdown("### Upload CSV File")
        st.info("""
        📋 **File Requirements:**
        - CSV format with headers
        - Must contain at least one text column with messages
        - Maximum file size: 200MB
        - Supported encodings: UTF-8, Latin-1
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Select a CSV file containing messages to classify"
        )
        
        if uploaded_file is not None:
            try:
                # Try reading with different encodings
                try:
                    df = pd.read_csv(uploaded_file)
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                
                st.success(f"✅ File loaded successfully: {len(df)} rows, {len(df.columns)} columns")
                
                # Show preview
                st.markdown("#### 📊 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Column selection
                st.markdown("#### 🎯 Select Text Column")
                text_column = st.selectbox(
                    "Which column contains the messages?",
                    options=df.columns.tolist(),
                    help="Select the column containing the text messages to classify"
                )
                
                # Show sample from selected column
                if text_column:
                    st.markdown("##### Sample Messages")
                    samples = df[text_column].dropna().head(3)
                    for i, sample in enumerate(samples, 1):
                        st.text(f"{i}. {str(sample)[:100]}...")
                    
                    # Process button
                    if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                        st.markdown("---")
                        st.markdown("### 🔄 Processing Messages...")
                        
                        start_time = time.time()
                        results_df = process_batch(df, text_column, model, vectorizer, threshold)
                        processing_time = time.time() - start_time
                        
                        # Merge results with original data
                        final_df = df.copy()
                        final_df['prediction'] = results_df['prediction']
                        final_df['spam_probability'] = results_df['spam_probability']
                        if include_confidence:
                            final_df['confidence'] = results_df['confidence']
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### ✅ Processing Complete!")
                        
                        # Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        total_messages = len(results_df)
                        spam_count = (results_df['prediction'] == 'spam').sum()
                        ham_count = (results_df['prediction'] == 'ham').sum()
                        error_count = (results_df['prediction'] == 'error').sum()
                        
                        with col1:
                            st.metric("Total Processed", total_messages)
                        with col2:
                            st.metric("Spam Detected", spam_count, 
                                    delta=f"{spam_count/total_messages*100:.1f}%")
                        with col3:
                            st.metric("Ham Detected", ham_count,
                                    delta=f"{ham_count/total_messages*100:.1f}%")
                        with col4:
                            st.metric("Processing Time", f"{processing_time:.2f}s")
                        
                        # Visualization
                        st.markdown("### 📊 Results Visualization")
                        
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            # Pie chart
                            fig_pie = px.pie(
                                values=[spam_count, ham_count],
                                names=['Spam', 'Ham'],
                                title='Classification Distribution',
                                color_discrete_map={'Spam': '#FF6B6B', 'Ham': '#4ECDC4'}
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with viz_col2:
                            # Confidence histogram
                            fig_hist = px.histogram(
                                results_df,
                                x='confidence',
                                nbins=20,
                                title='Confidence Score Distribution',
                                labels={'confidence': 'Confidence Score', 'count': 'Number of Messages'}
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Probability distribution
                        if include_probabilities:
                            fig_scatter = px.scatter(
                                results_df,
                                x='ham_probability',
                                y='spam_probability',
                                color='prediction',
                                title='Probability Distribution',
                                labels={'ham_probability': 'Ham Probability', 
                                       'spam_probability': 'Spam Probability'},
                                color_discrete_map={'spam': '#FF6B6B', 'ham': '#4ECDC4', 'error': '#FFD93D'}
                            )
                            fig_scatter.add_shape(
                                type="line",
                                x0=0, y0=threshold, x1=1, y1=threshold,
                                line=dict(color="red", width=2, dash="dash"),
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Results table
                        st.markdown("### 📋 Detailed Results")
                        
                        # Filter options
                        filter_col1, filter_col2 = st.columns(2)
                        with filter_col1:
                            show_filter = st.selectbox(
                                "Show:",
                                ["All", "Spam only", "Ham only", "Errors only"]
                            )
                        
                        # Apply filter
                        display_df = final_df.copy()
                        if show_filter == "Spam only":
                            display_df = display_df[display_df['prediction'] == 'spam']
                        elif show_filter == "Ham only":
                            display_df = display_df[display_df['prediction'] == 'ham']
                        elif show_filter == "Errors only":
                            display_df = display_df[display_df['prediction'] == 'error']
                        
                        st.dataframe(display_df, use_container_width=True, height=400)
                        
                        # Download results
                        st.markdown("### 💾 Download Results")
                        
                        csv = final_df.to_csv(index=False)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        st.download_button(
                            label="📥 Download Processed CSV",
                            data=csv,
                            file_name=f'spam_classification_results_{timestamp}.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
                        
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.info("Please ensure your file is a valid CSV with UTF-8 or Latin-1 encoding")
    
    with tab2:
        st.markdown("### 🎲 Sample Data Generator")
        st.info("Generate sample data to test the batch processing functionality")
        
        # Generate sample button
        if st.button("📝 Generate Sample CSV", use_container_width=True):
            sample_df = create_sample_csv()
            
            st.success("✅ Sample data generated!")
            st.dataframe(sample_df, use_container_width=True)
            
            # Download sample
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv,
                file_name='sample_messages.csv',
                mime='text/csv',
                use_container_width=True
            )
            
            st.markdown("---")
            st.info("💡 **Next Steps:**\n1. Download the sample CSV\n2. Go to 'Upload & Process' tab\n3. Upload the file and process it")
        
        # Instructions
        st.markdown("""
        ### 📖 How to Use Batch Processing
        
        1. **Prepare your CSV file** with messages in one column
        2. **Upload the file** using the file uploader
        3. **Select the text column** containing messages
        4. **Click 'Start Processing'** to classify all messages
        5. **Review results** in charts and tables
        6. **Download results** as CSV with predictions added
        
        ### 📊 Output Columns
        
        The processed CSV will include:
        - **prediction**: 'spam' or 'ham' classification
        - **spam_probability**: Probability of being spam (0-1)
        - **confidence**: Model's confidence in the prediction (0-1)
        
        ### 🎯 Tips for Best Results
        
        - Ensure messages are in plain text format
        - Remove any HTML tags or special formatting
        - One message per row in the CSV
        - Use UTF-8 encoding for international characters
        """)

if __name__ == "__main__":
    main()