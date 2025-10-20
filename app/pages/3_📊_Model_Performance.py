#!/usr/bin/env python3
"""
Model Performance Page - Visualize model metrics and performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from wordcloud import WordCloud
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Model Performance - Spam Classifier",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_metrics():
    """Load all available metrics"""
    metrics = {}
    reports_dir = Path('reports')
    
    if not reports_dir.exists():
        return metrics
    
    # Load benchmark results
    benchmark_file = reports_dir / 'benchmark_results.json'
    if benchmark_file.exists():
        with open(benchmark_file, 'r') as f:
            metrics['benchmark'] = json.load(f)
    
    # Load optimal threshold
    threshold_file = reports_dir / 'optimal_threshold.json'
    if threshold_file.exists():
        with open(threshold_file, 'r') as f:
            metrics['threshold'] = json.load(f)
    
    # Load regularization analysis
    reg_file = reports_dir / 'regularization_analysis.json'
    if reg_file.exists():
        with open(reg_file, 'r') as f:
            metrics['regularization'] = json.load(f)
    
    # Load cost analysis
    cost_file = reports_dir / 'cost_sensitive_analysis.json'
    if cost_file.exists():
        with open(cost_file, 'r') as f:
            metrics['cost'] = json.load(f)
    
    # Load LinearSVC comparison
    svc_file = reports_dir / 'linearsvc_comparison.json'
    if svc_file.exists():
        with open(svc_file, 'r') as f:
            metrics['svc_comparison'] = json.load(f)
    
    return metrics

@st.cache_data
def load_pr_roc_data():
    """Load PR and ROC curve data"""
    reports_dir = Path('reports')
    
    # Try to find curve data files
    pr_files = list(reports_dir.glob('*pr_curve*.json'))
    roc_files = list(reports_dir.glob('*roc_curve*.json'))
    
    pr_data = None
    roc_data = None
    
    if pr_files:
        with open(pr_files[0], 'r') as f:
            pr_data = json.load(f)
    
    if roc_files:
        with open(roc_files[0], 'r') as f:
            roc_data = json.load(f)
    
    return pr_data, roc_data

def create_confusion_matrix_plot(cm_data):
    """Create an interactive confusion matrix plot"""
    if not cm_data:
        cm = np.array([[850, 50], [30, 70]])  # Default example
    else:
        cm = np.array(cm_data)
    
    # Calculate percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=['Predicted Ham', 'Predicted Spam'],
        y=['Actual Ham', 'Actual Spam'],
        text=cm,
        texttemplate='%{text}',
        colorscale='RdBu_r',
        showscale=True,
        colorbar=dict(title="Percentage")
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=400
    )
    
    return fig

def create_pr_curve(pr_data=None):
    """Create Precision-Recall curve"""
    fig = go.Figure()
    
    if pr_data:
        fig.add_trace(go.Scatter(
            x=pr_data.get('recall', []),
            y=pr_data.get('precision', []),
            mode='lines',
            name=f'PR Curve (AUC = {pr_data.get("auc", 0):.3f})',
            line=dict(color='#FF6B6B', width=2)
        ))
    else:
        # Generate example curve
        recall = np.linspace(0, 1, 100)
        precision = 0.9 - 0.3 * recall + 0.1 * np.random.randn(100) * 0.1
        precision = np.clip(precision, 0, 1)
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name='PR Curve (Example)',
            line=dict(color='#FF6B6B', width=2)
        ))
    
    # Add baseline
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0.5, 0.5],
        mode='lines',
        name='Baseline',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    # Add target lines
    fig.add_hline(y=0.9, line_dash="dot", line_color="green", 
                  annotation_text="Precision Target (0.90)")
    fig.add_vline(x=0.93, line_dash="dot", line_color="blue",
                  annotation_text="Recall Target (0.93)")
    
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def create_roc_curve(roc_data=None):
    """Create ROC curve"""
    fig = go.Figure()
    
    if roc_data:
        fig.add_trace(go.Scatter(
            x=roc_data.get('fpr', []),
            y=roc_data.get('tpr', []),
            mode='lines',
            name=f'ROC Curve (AUC = {roc_data.get("auc", 0):.3f})',
            line=dict(color='#4ECDC4', width=2)
        ))
    else:
        # Generate example curve
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr) + 0.1 * np.random.randn(100) * 0.05
        tpr = np.clip(tpr, 0, 1)
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name='ROC Curve (Example)',
            line=dict(color='#4ECDC4', width=2)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def create_feature_importance_chart(model, vectorizer, n_features=20):
    """Create feature importance chart"""
    try:
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps.get('classifier', model.named_steps.get('model'))
            vectorizer = model.named_steps.get('vectorizer')
        else:
            classifier = model
        
        if not hasattr(classifier, 'coef_'):
            return None
        
        feature_names = vectorizer.get_feature_names_out()
        coef = classifier.coef_[0]
        
        # Get top positive and negative features
        top_positive_idx = np.argsort(coef)[-n_features//2:]
        top_negative_idx = np.argsort(coef)[:n_features//2]
        
        top_features = []
        for idx in top_positive_idx:
            top_features.append({'feature': feature_names[idx], 'weight': coef[idx], 'type': 'spam'})
        for idx in top_negative_idx:
            top_features.append({'feature': feature_names[idx], 'weight': coef[idx], 'type': 'ham'})
        
        df = pd.DataFrame(top_features)
        df = df.sort_values('weight')
        
        fig = px.bar(
            df,
            x='weight',
            y='feature',
            orientation='h',
            color='type',
            color_discrete_map={'spam': '#FF6B6B', 'ham': '#4ECDC4'},
            title=f'Top {n_features} Feature Importance',
            labels={'weight': 'Feature Weight', 'feature': 'Feature'}
        )
        
        fig.update_layout(height=600)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating feature importance chart: {str(e)}")
        return None

def main():
    st.title("📊 Model Performance Dashboard")
    st.markdown("Comprehensive analysis of spam classifier performance")
    
    # Load metrics
    metrics = load_metrics()
    pr_data, roc_data = load_pr_roc_data()
    
    if not metrics:
        st.warning("⚠️ No performance metrics found. Please run model evaluation first.")
        st.info("Run `npm run benchmark` to generate performance metrics")
        return
    
    # Performance Overview
    st.markdown("## 🎯 Performance Overview")
    
    if 'benchmark' in metrics and 'best_configs' in metrics['benchmark']:
        # Display best model metrics
        best_configs = metrics['benchmark']['best_configs']
        
        tabs = st.tabs([f"Phase {i+1}" for i in range(len(best_configs))])
        
        for i, (phase_key, config) in enumerate(best_configs.items()):
            with tabs[i]:
                if config:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Precision",
                            f"{config['precision']:.3f}",
                            delta="Target: ≥0.90",
                            delta_color="normal" if config['precision'] >= 0.90 else "inverse"
                        )
                    
                    with col2:
                        st.metric(
                            "Recall",
                            f"{config['recall']:.3f}",
                            delta="Target: ≥0.93",
                            delta_color="normal" if config['recall'] >= 0.93 else "inverse"
                        )
                    
                    with col3:
                        st.metric(
                            "F1-Score",
                            f"{config['f1_score']:.3f}"
                        )
                    
                    with col4:
                        st.metric(
                            "Accuracy",
                            f"{config.get('accuracy', 0):.3f}"
                        )
                    
                    # Confusion Matrix
                    if 'confusion_matrix' in config:
                        st.plotly_chart(
                            create_confusion_matrix_plot(config['confusion_matrix']),
                            use_container_width=True
                        )
    
    # Performance Curves
    st.markdown("## 📈 Performance Curves")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_pr_curve(pr_data), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_roc_curve(roc_data), use_container_width=True)
    
    # Model Comparison
    if 'benchmark' in metrics and 'comparison' in metrics['benchmark']:
        st.markdown("## 🔄 Model Evolution")
        
        comparison_df = pd.DataFrame(metrics['benchmark']['comparison'])
        
        # Create comparison chart
        fig = go.Figure()
        
        metrics_to_plot = ['precision', 'recall', 'f1_score', 'accuracy']
        colors = ['#FF6B6B', '#4ECDC4', '#95E77E', '#FFD93D']
        
        for metric, color in zip(metrics_to_plot, colors):
            if metric in comparison_df.columns:
                fig.add_trace(go.Scatter(
                    x=comparison_df['phase'],
                    y=comparison_df[metric],
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=color, width=2),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title="Performance Metrics Across Development Phases",
            xaxis_title="Development Phase",
            yaxis_title="Score",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.markdown("### 📋 Detailed Comparison")
        
        display_df = comparison_df.copy()
        for col in ['precision', 'recall', 'f1_score', 'accuracy']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Threshold Analysis
    if 'threshold' in metrics:
        st.markdown("## 🎚️ Threshold Analysis")
        
        threshold_data = metrics['threshold']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if 'sweep_results' in threshold_data:
                sweep_df = pd.DataFrame(threshold_data['sweep_results'])
                
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
                
                # Mark optimal threshold
                if 'optimal_configuration' in threshold_data:
                    optimal = threshold_data['optimal_configuration']
                    fig.add_vline(
                        x=optimal['threshold'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Optimal: {optimal['threshold']:.2f}"
                    )
                
                fig.update_layout(
                    title="Metrics vs Classification Threshold",
                    xaxis_title="Threshold",
                    yaxis_title="Score",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'optimal_configuration' in threshold_data:
                optimal = threshold_data['optimal_configuration']
                
                st.markdown("### ⚡ Optimal Threshold")
                st.metric("Threshold", f"{optimal['threshold']:.3f}")
                st.metric("Precision", f"{optimal['precision']:.3f}")
                st.metric("Recall", f"{optimal['recall']:.3f}")
                st.metric("F1-Score", f"{optimal['f1']:.3f}")
    
    # Cost Analysis
    if 'cost' in metrics:
        st.markdown("## 💰 Cost-Sensitive Analysis")
        
        cost_data = metrics['cost']
        
        if 'scenario_comparison' in cost_data:
            scenarios_df = pd.DataFrame(cost_data['scenario_comparison'])
            
            fig = px.bar(
                scenarios_df,
                x='scenario',
                y='total_cost',
                title='Total Cost by Scenario',
                labels={'total_cost': 'Total Cost ($)', 'scenario': 'Scenario'},
                color='total_cost',
                color_continuous_scale='RdYlGn_r'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        if 'best_configuration' in cost_data:
            best = cost_data['best_configuration']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Optimal C Value", f"{best.get('C', 'N/A')}")
            with col2:
                st.metric("Total Cost", f"${best.get('total_cost', 0):.2f}")
            with col3:
                cost_reduction = cost_data.get('cost_reduction_percentage', 0)
                st.metric("Cost Reduction", f"{cost_reduction:.1f}%")
    
    # Model Comparison
    if 'svc_comparison' in metrics:
        st.markdown("## 🤖 Model Algorithm Comparison")
        
        models_data = metrics['svc_comparison']['models']
        
        # Create comparison dataframe
        comparison_list = []
        for model_name, model_metrics in models_data.items():
            comparison_list.append({
                'Model': model_name.replace('_', ' ').title(),
                'Precision': model_metrics['precision'],
                'Recall': model_metrics['recall'],
                'F1-Score': model_metrics['f1'],
                'Training Time (s)': model_metrics['train_time'],
                'Speed (samples/sec)': model_metrics['predictions_per_sec']
            })
        
        comparison_df = pd.DataFrame(comparison_list)
        
        # Create comparison chart
        fig = px.bar(
            comparison_df,
            x='Model',
            y=['Precision', 'Recall', 'F1-Score'],
            title='Model Performance Comparison',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Speed comparison
        fig_speed = px.bar(
            comparison_df,
            x='Model',
            y='Speed (samples/sec)',
            title='Inference Speed Comparison',
            color='Speed (samples/sec)',
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig_speed, use_container_width=True)
    
    # Export Options
    st.markdown("## 💾 Export Performance Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Metrics as JSON", use_container_width=True):
            json_str = json.dumps(metrics, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="model_performance_metrics.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📈 Export Charts as HTML", use_container_width=True):
            st.info("Chart export will be available in the next update")
    
    with col3:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            st.info("PDF generation will be available in the next update")

if __name__ == "__main__":
    main()