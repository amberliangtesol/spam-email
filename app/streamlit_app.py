#!/usr/bin/env python3
"""
Streamlit Dashboard for Spam Email Classifier
Phase 4: Interactive Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure page
st.set_page_config(
    page_title="Spam Classifier Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/spam-classifier',
        'Report a bug': 'https://github.com/yourusername/spam-classifier/issues',
        'About': "# Spam Email Classifier\n\nAn advanced ML-powered spam detection system."
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background-color: #FF6B6B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border-radius: 0.3rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #FF5252;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and vectorizer"""
    try:
        model_dir = Path('models')
        
        # Try to find the latest model
        model_files = list(model_dir.glob('spam_classifier*.pkl'))
        if not model_files:
            model_files = list(model_dir.glob('*precision*.pkl'))
        if not model_files:
            model_files = list(model_dir.glob('*.pkl'))
            
        if not model_files:
            return None, None, None
        
        # Sort by modification time and get the latest
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        model_path = model_files[0]
        
        # Find corresponding vectorizer
        vectorizer_files = list(model_dir.glob('*vectorizer*.pkl'))
        if not vectorizer_files:
            # Try to extract vectorizer from pipeline
            model = joblib.load(model_path)
            if hasattr(model, 'named_steps'):
                return model, model.named_steps.get('vectorizer'), model_path.name
            return model, None, model_path.name
        
        vectorizer_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        vectorizer_path = vectorizer_files[0]
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        return model, vectorizer, model_path.name
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

@st.cache_data
def load_config():
    """Load the latest configuration"""
    try:
        config_path = Path('configs/precision_optimized_config.json')
        if not config_path.exists():
            config_path = Path('configs/default_config.json')
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return {}

@st.cache_data
def load_metrics():
    """Load model performance metrics"""
    try:
        reports_dir = Path('reports')
        
        # Try to load benchmark results first
        benchmark_file = reports_dir / 'benchmark_results.json'
        if benchmark_file.exists():
            with open(benchmark_file, 'r') as f:
                return json.load(f)
        
        # Otherwise load any available metrics
        metrics_files = list(reports_dir.glob('*metrics*.json'))
        if metrics_files:
            with open(metrics_files[0], 'r') as f:
                return json.load(f)
        
        return {}
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
        return {}

def display_model_info(model_name):
    """Display model information in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Information")
    
    if model_name:
        st.sidebar.success(f"✅ Model Loaded")
        st.sidebar.text(f"Version: {model_name}")
        
        # Get model stats
        model_path = Path('models') / model_name
        if model_path.exists():
            file_size = model_path.stat().st_size / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
            st.sidebar.text(f"Size: {file_size:.2f} MB")
            st.sidebar.text(f"Modified: {mod_time:%Y-%m-%d %H:%M}")
    else:
        st.sidebar.error("❌ No Model Loaded")
        st.sidebar.info("Please train a model first using:\n`npm run train`")

def display_quick_metrics(metrics):
    """Display quick metrics in sidebar"""
    if metrics and 'best_configs' in metrics:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎯 Best Performance")
        
        for phase, config in metrics['best_configs'].items():
            if config:
                phase_name = phase.replace('phase', 'Phase ')
                with st.sidebar.expander(f"{phase_name.title()}", expanded=False):
                    st.metric("Precision", f"{config.get('precision', 0):.3f}")
                    st.metric("Recall", f"{config.get('recall', 0):.3f}")
                    st.metric("F1-Score", f"{config.get('f1_score', 0):.3f}")

def main():
    """Main application"""
    # Header
    st.markdown('<h1 class="main-header">📧 Spam Email Classifier Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced ML-Powered Spam Detection System</p>', unsafe_allow_html=True)
    
    # Load model and data
    model, vectorizer, model_name = load_model_and_vectorizer()
    config = load_config()
    metrics = load_metrics()
    
    # Display model info in sidebar
    display_model_info(model_name)
    display_quick_metrics(metrics)
    
    # Navigation info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")
    st.sidebar.info(
        "Use the sidebar to navigate between different pages:\n\n"
        "• **Live Prediction** - Classify single messages\n"
        "• **Batch Processing** - Process multiple messages\n"
        "• **Model Performance** - View metrics and charts\n"
        "• **Threshold Tuning** - Adjust decision threshold\n"
        "• **Data Explorer** - Explore training data"
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🚀 Quick Start", "📊 System Status", "ℹ️ About"])
    
    with tab1:
        st.markdown("## Dashboard Overview")
        
        if model is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            # Display key metrics if available
            if metrics and 'best_configs' in metrics:
                best_phase3 = metrics['best_configs'].get('phase3', {})
                
                with col1:
                    st.metric(
                        label="🎯 Precision",
                        value=f"{best_phase3.get('precision', 0):.2%}",
                        delta="Target: ≥90%",
                        delta_color="normal"
                    )
                
                with col2:
                    st.metric(
                        label="📊 Recall",
                        value=f"{best_phase3.get('recall', 0):.2%}",
                        delta="Target: ≥93%",
                        delta_color="normal"
                    )
                
                with col3:
                    st.metric(
                        label="⚖️ F1-Score",
                        value=f"{best_phase3.get('f1_score', 0):.2%}",
                        delta="Balanced metric"
                    )
                
                with col4:
                    accuracy = best_phase3.get('accuracy', 0)
                    st.metric(
                        label="✅ Accuracy",
                        value=f"{accuracy:.2%}",
                        delta="Overall performance"
                    )
            
            # Phase comparison chart
            st.markdown("### Model Evolution Across Phases")
            if metrics and 'comparison' in metrics:
                comparison_df = pd.DataFrame(metrics['comparison'])
                
                fig = go.Figure()
                
                phases = comparison_df['phase'].tolist()
                precision = comparison_df['precision'].tolist()
                recall = comparison_df['recall'].tolist()
                f1 = comparison_df['f1_score'].tolist()
                
                fig.add_trace(go.Bar(name='Precision', x=phases, y=precision, marker_color='#FF6B6B'))
                fig.add_trace(go.Bar(name='Recall', x=phases, y=recall, marker_color='#4ECDC4'))
                fig.add_trace(go.Bar(name='F1-Score', x=phases, y=f1, marker_color='#95E77E'))
                
                fig.update_layout(
                    title="Performance Metrics by Phase",
                    xaxis_title="Development Phase",
                    yaxis_title="Score",
                    barmode='group',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            if metrics and 'recommendations' in metrics:
                st.markdown("### 💡 System Recommendations")
                for rec in metrics['recommendations'][:3]:
                    st.info(f"• {rec}")
        else:
            st.warning("⚠️ No trained model found. Please train a model first.")
            st.markdown("""
            ### Getting Started
            1. Run `npm run train` to train the baseline model
            2. Run `npm run train:recall` for recall-optimized model
            3. Run `npm run train:precision` for precision-optimized model
            4. Refresh this dashboard to see results
            """)
    
    with tab2:
        st.markdown("## 🚀 Quick Start Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### For Users
            1. **Live Prediction**: Navigate to "1_Live_Prediction" to classify single messages
            2. **Batch Processing**: Use "2_Batch_Processing" for CSV file classification
            3. **View Performance**: Check "3_Model_Performance" for detailed metrics
            4. **Tune Threshold**: Adjust decision threshold in "4_Threshold_Tuning"
            5. **Explore Data**: Analyze training data in "5_Data_Explorer"
            """)
        
        with col2:
            st.markdown("""
            ### For Developers
            1. **Train Models**: `npm run train:precision`
            2. **Run Benchmarks**: `npm run benchmark`
            3. **Optimize Threshold**: `npm run sweep:threshold`
            4. **Compare Models**: `npm run compare:models`
            5. **Deploy to Cloud**: `streamlit deploy`
            """)
        
        st.markdown("---")
        st.markdown("""
        ### 📚 Documentation
        - [API Documentation](/docs) - REST API endpoints
        - [Model Training Guide](./README.md) - How to train custom models
        - [Configuration Reference](./configs/) - Configuration options
        - [GitHub Repository](https://github.com/yourusername/spam-classifier) - Source code
        """)
    
    with tab3:
        st.markdown("## 📊 System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Model Status")
            if model:
                st.success("✅ Model Loaded Successfully")
                st.text(f"Type: {type(model).__name__}")
                if hasattr(model, 'n_features_in_'):
                    st.text(f"Features: {model.n_features_in_:,}")
            else:
                st.error("❌ Model Not Loaded")
        
        with col2:
            st.markdown("### Vectorizer Status")
            if vectorizer:
                st.success("✅ Vectorizer Loaded")
                if hasattr(vectorizer, 'vocabulary_'):
                    st.text(f"Vocabulary: {len(vectorizer.vocabulary_):,} words")
            else:
                st.warning("⚠️ Using Pipeline Vectorizer")
        
        with col3:
            st.markdown("### Configuration")
            if config:
                st.success("✅ Config Loaded")
                st.text(f"Version: {config.get('version', 'unknown')}")
                st.text(f"Phase: {config.get('phase', 'unknown')}")
            else:
                st.error("❌ Config Not Found")
        
        # File system check
        st.markdown("### File System Check")
        
        paths_to_check = {
            'Models Directory': Path('models'),
            'Reports Directory': Path('reports'),
            'Configs Directory': Path('configs'),
            'Data Directory': Path('datasets')
        }
        
        for name, path in paths_to_check.items():
            if path.exists():
                file_count = len(list(path.glob('*')))
                st.success(f"✅ {name}: {file_count} files")
            else:
                st.error(f"❌ {name}: Not found")
    
    with tab4:
        st.markdown("## ℹ️ About This Project")
        
        st.markdown("""
        ### Spam Email Classifier
        
        This is an advanced machine learning system for spam email classification, developed through 
        four iterative phases following OpenSpec methodology:
        
        1. **Phase 1**: Baseline Classifier - Basic TF-IDF and Logistic Regression
        2. **Phase 2**: Recall Improvement - Optimized for catching more spam (Recall ≥ 93%)
        3. **Phase 3**: Precision Recovery - Balanced precision and recall (Precision ≥ 90%)
        4. **Phase 4**: Interactive Dashboard - This Streamlit application
        
        ### Key Features
        - **Real-time Classification**: Instant spam/ham prediction
        - **Batch Processing**: Handle multiple emails efficiently
        - **Threshold Tuning**: Adjustable decision boundaries
        - **Performance Monitoring**: Comprehensive metrics and visualizations
        - **Data Exploration**: Interactive dataset analysis
        
        ### Technology Stack
        - **ML Framework**: scikit-learn
        - **Dashboard**: Streamlit
        - **Visualization**: Plotly, Matplotlib, Seaborn
        - **API**: FastAPI
        - **Deployment**: Streamlit Cloud / Docker
        
        ### Performance Targets
        - Precision: ≥ 90% (minimize false positives)
        - Recall: ≥ 93% (minimize false negatives)
        - F1-Score: Balanced performance metric
        - Inference: < 100ms per message
        
        ### Version Information
        - Dashboard Version: 1.0.0
        - Model Version: {model_name if model_name else 'Not Loaded'}
        - Last Updated: {datetime.now():%Y-%m-%d}
        """)
        
        st.markdown("---")
        st.markdown("""
        ### 👥 Contributing
        
        We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.
        
        ### 📝 License
        
        This project is licensed under the MIT License.
        
        ### 🙏 Acknowledgments
        
        - OpenSpec methodology for structured development
        - scikit-learn community for ML tools
        - Streamlit team for the amazing framework
        """)

if __name__ == "__main__":
    main()