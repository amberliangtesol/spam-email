#!/usr/bin/env python3
"""
Data Explorer Page - Explore and analyze the training dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Data Explorer - Spam Classifier",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data
def load_dataset():
    """Load the training dataset"""
    try:
        # Try different possible locations
        paths_to_try = [
            Path('datasets/sms_spam_no_header.csv'),
            Path('datasets/processed/train_data.csv'),
            Path('datasets/spam.csv')
        ]
        
        for path in paths_to_try:
            if path.exists():
                # Try to load with headers first
                try:
                    df = pd.read_csv(path)
                    if 'label' not in df.columns and 'text' not in df.columns:
                        # Try without headers
                        df = pd.read_csv(path, header=None, names=['label', 'text'])
                except:
                    df = pd.read_csv(path, header=None, names=['label', 'text'])
                
                # Convert labels if necessary
                if df['label'].dtype == 'object':
                    if df['label'].str.contains('spam|ham').any():
                        df['label'] = (df['label'] == 'spam').astype(int)
                    else:
                        # Assume first column is label
                        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
                
                # Add derived features
                df['message_length'] = df['text'].str.len()
                df['word_count'] = df['text'].str.split().str.len()
                df['exclamation_count'] = df['text'].str.count('!')
                df['question_count'] = df['text'].str.count(r'\?')
                df['uppercase_ratio'] = df['text'].apply(
                    lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
                )
                
                return df
        
        return None
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def get_word_frequencies(texts, n_words=20):
    """Get word frequencies from texts"""
    all_words = []
    for text in texts:
        # Basic text cleaning
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        words = text.split()
        all_words.extend(words)
    
    # Count frequencies
    word_freq = Counter(all_words)
    
    # Remove common stop words (basic list)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'been', 'be',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'can', 'could', 'i', 'you', 'he',
                  'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
                  'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
                  'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                  'so', 'than', 'too', 'very', 'this', 'that', 'these', 'those'}
    
    for word in stop_words:
        word_freq.pop(word, None)
    
    return word_freq.most_common(n_words)

def create_wordcloud(texts, title="Word Cloud"):
    """Create a word cloud from texts"""
    # Combine all texts
    combined_text = ' '.join(str(text) for text in texts)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(combined_text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    return fig

def main():
    st.title("🔍 Data Explorer")
    st.markdown("Explore and analyze the training dataset")
    
    # Load data
    df = load_dataset()
    
    if df is None:
        st.error("❌ No dataset found. Please ensure training data is available.")
        st.info("Expected location: `datasets/sms_spam_no_header.csv`")
        return
    
    # Dataset overview
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_messages = len(df)
    spam_count = (df['label'] == 1).sum()
    ham_count = (df['label'] == 0).sum()
    spam_ratio = spam_count / total_messages * 100
    
    with col1:
        st.metric("Total Messages", f"{total_messages:,}")
    
    with col2:
        st.metric("Spam Messages", f"{spam_count:,}", delta=f"{spam_ratio:.1f}%")
    
    with col3:
        st.metric("Ham Messages", f"{ham_count:,}", delta=f"{100-spam_ratio:.1f}%")
    
    with col4:
        st.metric("Class Imbalance", f"{max(spam_count, ham_count) / min(spam_count, ham_count):.1f}:1")
    
    # Class distribution
    st.markdown("### 📈 Class Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig_pie = px.pie(
            values=[ham_count, spam_count],
            names=['Ham', 'Spam'],
            title='Message Distribution',
            color_discrete_map={'Ham': '#4ECDC4', 'Spam': '#FF6B6B'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Message length distribution
        fig_box = px.box(
            df,
            x='label',
            y='message_length',
            title='Message Length by Class',
            labels={'label': 'Class', 'message_length': 'Character Count'},
            color='label',
            color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'}
        )
        fig_box.update_xaxis(ticktext=['Ham', 'Spam'], tickvals=[0, 1])
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Statistical analysis
    st.markdown("### 📊 Statistical Analysis")
    
    # Calculate statistics by class
    stats_df = df.groupby('label').agg({
        'message_length': ['mean', 'std', 'min', 'max'],
        'word_count': ['mean', 'std', 'min', 'max'],
        'exclamation_count': 'mean',
        'question_count': 'mean',
        'uppercase_ratio': 'mean'
    }).round(2)
    
    stats_df.index = ['Ham', 'Spam']
    st.dataframe(stats_df, use_container_width=True)
    
    # Feature distributions
    st.markdown("### 📈 Feature Distributions")
    
    feature_tabs = st.tabs(["Length", "Punctuation", "Capitalization"])
    
    with feature_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                df,
                x='message_length',
                color='label',
                title='Message Length Distribution',
                nbins=30,
                color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'},
                labels={'label': 'Class', 'message_length': 'Character Count'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_hist2 = px.histogram(
                df,
                x='word_count',
                color='label',
                title='Word Count Distribution',
                nbins=30,
                color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'},
                labels={'label': 'Class', 'word_count': 'Number of Words'}
            )
            st.plotly_chart(fig_hist2, use_container_width=True)
    
    with feature_tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_exc = px.histogram(
                df,
                x='exclamation_count',
                color='label',
                title='Exclamation Marks Distribution',
                nbins=20,
                color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'}
            )
            st.plotly_chart(fig_exc, use_container_width=True)
        
        with col2:
            fig_quest = px.histogram(
                df,
                x='question_count',
                color='label',
                title='Question Marks Distribution',
                nbins=20,
                color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'}
            )
            st.plotly_chart(fig_quest, use_container_width=True)
    
    with feature_tabs[2]:
        fig_caps = px.box(
            df,
            x='label',
            y='uppercase_ratio',
            title='Uppercase Ratio by Class',
            color='label',
            color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'}
        )
        fig_caps.update_xaxis(ticktext=['Ham', 'Spam'], tickvals=[0, 1])
        st.plotly_chart(fig_caps, use_container_width=True)
    
    # Word analysis
    st.markdown("### 💬 Word Analysis")
    
    spam_texts = df[df['label'] == 1]['text']
    ham_texts = df[df['label'] == 0]['text']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Spam Words")
        spam_words = get_word_frequencies(spam_texts, 15)
        spam_words_df = pd.DataFrame(spam_words, columns=['Word', 'Frequency'])
        
        fig_spam = px.bar(
            spam_words_df,
            x='Frequency',
            y='Word',
            orientation='h',
            title='Most Common Spam Words',
            color='Frequency',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_spam, use_container_width=True)
    
    with col2:
        st.markdown("#### Top Ham Words")
        ham_words = get_word_frequencies(ham_texts, 15)
        ham_words_df = pd.DataFrame(ham_words, columns=['Word', 'Frequency'])
        
        fig_ham = px.bar(
            ham_words_df,
            x='Frequency',
            y='Word',
            orientation='h',
            title='Most Common Ham Words',
            color='Frequency',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_ham, use_container_width=True)
    
    # Word clouds
    st.markdown("### ☁️ Word Clouds")
    
    with st.spinner("Generating word clouds..."):
        col1, col2 = st.columns(2)
        
        with col1:
            fig_wc_spam = create_wordcloud(spam_texts, "Spam Word Cloud")
            st.pyplot(fig_wc_spam)
        
        with col2:
            fig_wc_ham = create_wordcloud(ham_texts, "Ham Word Cloud")
            st.pyplot(fig_wc_ham)
    
    # Sample messages
    st.markdown("### 📝 Sample Messages")
    
    sample_tab1, sample_tab2 = st.tabs(["Spam Samples", "Ham Samples"])
    
    with sample_tab1:
        st.markdown("#### Random Spam Messages")
        spam_samples = df[df['label'] == 1].sample(min(5, spam_count))
        for idx, row in spam_samples.iterrows():
            st.text(f"• {row['text'][:200]}...")
    
    with sample_tab2:
        st.markdown("#### Random Ham Messages")
        ham_samples = df[df['label'] == 0].sample(min(5, ham_count))
        for idx, row in ham_samples.iterrows():
            st.text(f"• {row['text'][:200]}...")
    
    # Search and filter
    st.markdown("### 🔎 Search & Filter")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("Search messages", placeholder="Enter keyword...")
    
    with col2:
        filter_class = st.selectbox("Filter by class", ["All", "Spam", "Ham"])
    
    with col3:
        sort_by = st.selectbox("Sort by", ["Message Length", "Word Count", "Exclamation Count"])
    
    # Apply filters
    filtered_df = df.copy()
    
    if search_term:
        filtered_df = filtered_df[filtered_df['text'].str.contains(search_term, case=False, na=False)]
    
    if filter_class == "Spam":
        filtered_df = filtered_df[filtered_df['label'] == 1]
    elif filter_class == "Ham":
        filtered_df = filtered_df[filtered_df['label'] == 0]
    
    # Sort
    sort_column = {
        "Message Length": "message_length",
        "Word Count": "word_count",
        "Exclamation Count": "exclamation_count"
    }[sort_by]
    
    filtered_df = filtered_df.sort_values(sort_column, ascending=False)
    
    st.info(f"Found {len(filtered_df)} messages matching criteria")
    
    # Display filtered results
    if len(filtered_df) > 0:
        display_df = filtered_df[['text', 'label', 'message_length', 'word_count']].head(20)
        display_df['label'] = display_df['label'].map({0: 'Ham', 1: 'Spam'})
        display_df.columns = ['Message', 'Class', 'Length', 'Words']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Export filtered data
        if st.button("📥 Export Filtered Data"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="filtered_messages.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()