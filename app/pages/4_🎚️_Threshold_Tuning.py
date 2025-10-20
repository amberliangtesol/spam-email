#!/usr/bin/env python3
"""
Threshold Tuning Page - Interactive threshold adjustment
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Threshold Tuning - Spam Classifier",
    page_icon="🎚️",
    layout="wide"
)

@st.cache_resource
def load_model_and_data():
    """Load model and test data"""
    try:
        # Try to use model_loader first
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        try:
            from model_loader import load_or_create_model
            model, vectorizer, model_name = load_or_create_model()
        except ImportError:
            # Fallback to loading from models directory
            model_dir = Path('models')
            model_files = list(model_dir.glob('*.pkl'))
            
            if not model_files:
                return None, None, None, None
            
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            model = joblib.load(model_files[0])
            vectorizer = None
        
        # Load test data
        data_dir = Path('datasets')
        X_test = None
        y_test = None
        
        # Try to load SMS spam dataset
        sms_file = data_dir / 'sms_spam_no_header.csv'
        if sms_file.exists():
            df = pd.read_csv(sms_file, header=None, names=['label', 'message'])
            # Convert labels
            df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
            
            # Use last 20% as test data
            test_size = int(len(df) * 0.2)
            test_df = df.tail(test_size)
            
            X_test = test_df['message'].values
            y_test = test_df['label_num'].values
        else:
            # Try other test files
            test_files = list(data_dir.glob('*test*.csv'))
            if test_files:
                test_df = pd.read_csv(test_files[0])
                if 'text' in test_df.columns and 'label' in test_df.columns:
                    X_test = test_df['text'].values
                    y_test = test_df['label'].values
                    if y_test.dtype == 'object':
                        y_test = (y_test == 'spam').astype(int)
        
        return model, vectorizer, X_test, y_test
        
    except Exception as e:
        st.error(f"Error loading model/data: {str(e)}")
        return None, None, None, None

def calculate_metrics_at_threshold(y_true, y_proba, threshold):
    """Calculate metrics at a specific threshold"""
    y_pred = (y_proba >= threshold).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total_spam': int(tp + fn),
        'total_ham': int(tn + fp),
        'predicted_spam': int(tp + fp),
        'predicted_ham': int(tn + fn)
    }

def main():
    st.title("🎚️ Interactive Threshold Tuning")
    st.markdown("Fine-tune the classification threshold to optimize for your specific needs")
    
    # Load model and data
    model, vectorizer, X_test, y_test = load_model_and_data()
    
    if model is None:
        st.error("❌ No trained model found. Please train a model first.")
        st.stop()
    
    # Generate predictions
    if X_test is not None and y_test is not None:
        with st.spinner("Generating predictions..."):
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                st.error("Model doesn't support probability predictions")
                st.stop()
    else:
        # Generate synthetic data for demonstration
        st.warning("⚠️ No test data found. Using synthetic data for demonstration.")
        np.random.seed(42)
        n_samples = 1000
        y_test = np.random.binomial(1, 0.3, n_samples)
        y_proba = np.clip(y_test + np.random.normal(0, 0.3, n_samples), 0, 1)
    
    # Sidebar controls
    st.sidebar.markdown("### ⚙️ Threshold Settings")
    
    # Optimization target
    optimization_target = st.sidebar.selectbox(
        "Optimize for:",
        ["Balanced (F1)", "High Precision", "High Recall", "Custom"]
    )
    
    if optimization_target == "Custom":
        target_precision = st.sidebar.slider("Target Precision", 0.0, 1.0, 0.90, 0.01)
        target_recall = st.sidebar.slider("Target Recall", 0.0, 1.0, 0.93, 0.01)
    
    # Main threshold slider
    st.markdown("## 🎯 Adjust Classification Threshold")
    
    threshold = st.slider(
        "Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Messages with spam probability above this threshold are classified as spam"
    )
    
    # Calculate metrics at current threshold
    metrics = calculate_metrics_at_threshold(y_test, y_proba, threshold)
    
    # Display current metrics
    st.markdown("### 📊 Current Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta = "Target: ≥0.90" if metrics['precision'] < 0.90 else "✓ Target Met"
        st.metric("Precision", f"{metrics['precision']:.3f}", delta=delta)
    
    with col2:
        delta = "Target: ≥0.93" if metrics['recall'] < 0.93 else "✓ Target Met"
        st.metric("Recall", f"{metrics['recall']:.3f}", delta=delta)
    
    with col3:
        st.metric("F1-Score", f"{metrics['f1']:.3f}")
    
    with col4:
        st.metric("Threshold", f"{threshold:.3f}")
    
    # Confusion Matrix
    st.markdown("### 🔢 Confusion Matrix at Current Threshold")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Predictions")
        st.info(f"**Predicted Spam:** {metrics['predicted_spam']}")
        st.info(f"**Predicted Ham:** {metrics['predicted_ham']}")
        st.success(f"**True Positives:** {metrics['tp']}")
        st.error(f"**False Positives:** {metrics['fp']}")
        st.warning(f"**False Negatives:** {metrics['fn']}")
        st.success(f"**True Negatives:** {metrics['tn']}")
    
    with col2:
        # Create confusion matrix heatmap
        cm = np.array([[metrics['tn'], metrics['fp']], 
                      [metrics['fn'], metrics['tp']]])
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Ham', 'Predicted Spam'],
            y=['Actual Ham', 'Actual Spam'],
            text=cm,
            texttemplate='%{text}',
            colorscale='RdBu_r',
            showscale=True
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Threshold sweep visualization
    st.markdown("### 📈 Performance Across All Thresholds")
    
    # Calculate metrics for all thresholds
    thresholds = np.arange(0.01, 1.0, 0.01)
    sweep_results = []
    
    with st.spinner("Calculating metrics across thresholds..."):
        for t in thresholds:
            m = calculate_metrics_at_threshold(y_test, y_proba, t)
            sweep_results.append({
                'threshold': t,
                'precision': m['precision'],
                'recall': m['recall'],
                'f1': m['f1']
            })
    
    sweep_df = pd.DataFrame(sweep_results)
    
    # Create line plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sweep_df['threshold'],
        y=sweep_df['precision'],
        mode='lines',
        name='Precision',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=sweep_df['threshold'],
        y=sweep_df['recall'],
        mode='lines',
        name='Recall',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=sweep_df['threshold'],
        y=sweep_df['f1'],
        mode='lines',
        name='F1-Score',
        line=dict(color='#95E77E', width=2)
    ))
    
    # Add current threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Current: {threshold:.2f}"
    )
    
    # Add target lines
    fig.add_hline(y=0.90, line_dash="dot", line_color="gray", 
                  annotation_text="Precision Target")
    fig.add_hline(y=0.93, line_dash="dot", line_color="gray",
                  annotation_text="Recall Target")
    
    fig.update_layout(
        title="Metrics vs Threshold",
        xaxis_title="Threshold",
        yaxis_title="Score",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Find optimal thresholds
    st.markdown("### 🎯 Recommended Thresholds")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Max F1
        best_f1_idx = sweep_df['f1'].idxmax()
        best_f1_threshold = sweep_df.loc[best_f1_idx, 'threshold']
        best_f1_score = sweep_df.loc[best_f1_idx, 'f1']
        
        st.info(f"**Maximum F1-Score**")
        st.metric("Threshold", f"{best_f1_threshold:.3f}")
        st.metric("F1-Score", f"{best_f1_score:.3f}")
        
        if st.button("Apply Max F1", use_container_width=True):
            st.session_state.threshold = best_f1_threshold
            st.rerun()
    
    with col2:
        # Target precision
        precision_candidates = sweep_df[sweep_df['precision'] >= 0.90]
        if not precision_candidates.empty:
            best_recall_at_precision = precision_candidates.loc[precision_candidates['recall'].idxmax()]
            
            st.success(f"**Target Precision ≥ 0.90**")
            st.metric("Threshold", f"{best_recall_at_precision['threshold']:.3f}")
            st.metric("Recall", f"{best_recall_at_precision['recall']:.3f}")
            
            if st.button("Apply Precision Target", use_container_width=True):
                st.session_state.threshold = best_recall_at_precision['threshold']
                st.rerun()
        else:
            st.warning("No threshold meets precision target")
    
    with col3:
        # Target recall
        recall_candidates = sweep_df[sweep_df['recall'] >= 0.93]
        if not recall_candidates.empty:
            best_precision_at_recall = recall_candidates.loc[recall_candidates['precision'].idxmax()]
            
            st.success(f"**Target Recall ≥ 0.93**")
            st.metric("Threshold", f"{best_precision_at_recall['threshold']:.3f}")
            st.metric("Precision", f"{best_precision_at_recall['precision']:.3f}")
            
            if st.button("Apply Recall Target", use_container_width=True):
                st.session_state.threshold = best_precision_at_recall['threshold']
                st.rerun()
        else:
            st.warning("No threshold meets recall target")
    
    # Both targets
    both_targets = sweep_df[(sweep_df['precision'] >= 0.90) & (sweep_df['recall'] >= 0.93)]
    if not both_targets.empty:
        st.markdown("### ✅ Thresholds Meeting Both Targets")
        
        best_both = both_targets.loc[both_targets['f1'].idxmax()]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Optimal Threshold", f"{best_both['threshold']:.3f}")
        with col2:
            st.metric("Precision", f"{best_both['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{best_both['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{best_both['f1']:.3f}")
        
        if st.button("🎯 Apply Optimal Threshold", type="primary", use_container_width=True):
            st.session_state.threshold = best_both['threshold']
            st.rerun()
    else:
        st.warning("⚠️ No single threshold meets both precision (≥0.90) and recall (≥0.93) targets")
    
    # Export configuration
    st.markdown("### 💾 Export Threshold Configuration")
    
    config = {
        'threshold': threshold,
        'metrics': metrics,
        'optimization_target': optimization_target,
        'meets_precision_target': metrics['precision'] >= 0.90,
        'meets_recall_target': metrics['recall'] >= 0.93
    }
    
    import json
    config_json = json.dumps(config, indent=2)
    
    st.download_button(
        label="📥 Download Threshold Config",
        data=config_json,
        file_name=f"threshold_config_{threshold:.3f}.json",
        mime="application/json",
        use_container_width=True
    )

if __name__ == "__main__":
    main()