#!/usr/bin/env python3
"""
Credit Card Fraud Detection Dashboard
====================================

A Streamlit-based interactive dashboard for the credit card fraud detection system.
Provides real-time predictions, model performance visualization, and data exploration.

Author: [Your Name]
Date: [Current Date]
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pickle
import os
import warnings

# Machine Learning
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
    }
    .legitimate-alert {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

class FraudDetectionDashboard:
    """Streamlit dashboard for credit card fraud detection"""
    
    def __init__(self):
        self.models_dir = 'models'
        self.results_dir = 'results'
        self.data_path = 'creditcard.csv'
        
    def load_models(self):
        """Load trained models"""
        models = {}
        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                if file.endswith('_model.pkl'):
                    model_name = file.replace('_model.pkl', '').replace('_', ' ').title()
                    model_path = os.path.join(self.models_dir, file)
                    try:
                        models[model_name] = joblib.load(model_path)
                        st.success(f"✅ Loaded {model_name} model")
                    except Exception as e:
                        st.error(f"❌ Error loading {model_name}: {e}")
        return models
    
    def load_results(self):
        """Load model performance results"""
        results_path = os.path.join(self.results_dir, 'model_performance.csv')
        if os.path.exists(results_path):
            return pd.read_csv(results_path, index_col=0)
        return None
    
    def load_data(self):
        """Load sample data for demonstration"""
        try:
            if os.path.exists(self.data_path):
                # Load only a sample for faster processing
                df = pd.read_csv(self.data_path, nrows=10000)
                return df
            else:
                st.error("❌ Dataset not found. Please ensure 'creditcard.csv' is in the current directory.")
                return None
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return None
    
    def create_features(self, df):
        """Create engineered features"""
        df_copy = df.copy()
        
        # Amount-based features
        df_copy['Amount_Log'] = np.log1p(df_copy['Amount'])
        df_copy['Amount_Squared'] = df_copy['Amount'] ** 2
        
        # Time-based features
        df_copy['Hour'] = (df_copy['Time'] // 3600) % 24
        df_copy['Day'] = (df_copy['Time'] // 86400) % 7
        
        # Statistical features for V columns
        v_columns = [col for col in df_copy.columns if col.startswith('V')]
        df_copy['V_Mean'] = df_copy[v_columns].mean(axis=1)
        df_copy['V_Std'] = df_copy[v_columns].std(axis=1)
        df_copy['V_Max'] = df_copy[v_columns].max(axis=1)
        df_copy['V_Min'] = df_copy[v_columns].min(axis=1)
        
        # Interaction features
        df_copy['Amount_V1_Interaction'] = df_copy['Amount'] * df_copy['V1']
        df_copy['Amount_V2_Interaction'] = df_copy['Amount'] * df_copy['V2']
        
        return df_copy
    
    def predict_fraud(self, transaction_data, model, model_name):
        """Predict fraud for a transaction"""
        try:
            # Convert to DataFrame
            if isinstance(transaction_data, dict):
                df_transaction = pd.DataFrame([transaction_data])
            else:
                df_transaction = pd.DataFrame([transaction_data])
            
            # Apply feature engineering if needed
            if model_name == 'Logistic Regression':
                df_transaction = self.create_features(df_transaction)
            
            # Make prediction
            prediction = model.predict(df_transaction)[0]
            probability = model.predict_proba(df_transaction)[0, 1]
            
            return prediction, probability
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            return None, None
    
    def run_dashboard(self):
        """Run the main dashboard"""
        # Header
        st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("🔧 Dashboard Controls")
        
        # Load models
        st.sidebar.subheader("📊 Model Status")
        models = self.load_models()
        
        if not models:
            st.sidebar.error("❌ No trained models found. Please run the training script first.")
            st.info("💡 To get started, run: `python enhanced_credit_card_fraud_detection.py`")
            return
        
        # Model selection
        selected_model = st.sidebar.selectbox(
            "Select Model for Predictions:",
            list(models.keys()),
            index=0
        )
        
        # Load results
        results = self.load_results()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔮 Predictions", "📊 Analysis", "📈 Performance"])
        
        with tab1:
            self.show_home_tab(models, results)
        
        with tab2:
            self.show_predictions_tab(models, selected_model)
        
        with tab3:
            self.show_analysis_tab()
        
        with tab4:
            self.show_performance_tab(results)
    
    def show_home_tab(self, models, results):
        """Display home tab content"""
        st.header("🎯 Welcome to Credit Card Fraud Detection System")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Available Models", len(models))
        
        with col2:
            if results is not None:
                best_model = results['F1-Score'].idxmax()
                st.metric("Best Model", best_model)
            else:
                st.metric("Best Model", "N/A")
        
        with col3:
            if results is not None:
                best_f1 = results['F1-Score'].max()
                st.metric("Best F1-Score", f"{best_f1:.3f}")
            else:
                st.metric("Best F1-Score", "N/A")
        
        with col4:
            if results is not None:
                best_auc = results['AUC-ROC'].max()
                st.metric("Best AUC-ROC", f"{best_auc:.3f}")
            else:
                st.metric("Best AUC-ROC", "N/A")
        
        # Project description
        st.markdown("""
        ### 🚀 About This Project
        
        This is a comprehensive credit card fraud detection system that uses machine learning 
        to identify potentially fraudulent transactions. The system features:
        
        - **Multiple ML Models**: Logistic Regression, Random Forest, and XGBoost
        - **Advanced Feature Engineering**: 15+ engineered features for improved performance
        - **Class Imbalance Handling**: SMOTE technique for balanced training
        - **Real-time Predictions**: Instant fraud detection for new transactions
        - **Comprehensive Evaluation**: Detailed performance metrics and visualizations
        
        ### 📊 Dataset Information
        
        - **Total Transactions**: 284,807
        - **Features**: 30 numerical features (anonymized)
        - **Fraud Rate**: ~0.17% (highly imbalanced)
        - **Data Source**: Credit Card Fraud Detection Dataset
        
        ### 🔧 How to Use
        
        1. **Select a model** from the sidebar
        2. **Go to Predictions tab** to test new transactions
        3. **View Analysis tab** for data exploration
        4. **Check Performance tab** for model comparisons
        """)
        
        # Quick stats
        if results is not None:
            st.subheader("📈 Quick Performance Overview")
            
            # Performance comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            results[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(kind='bar', ax=ax)
            ax.set_title('Model Performance Comparison')
            ax.set_ylabel('Score')
            ax.set_xlabel('Models')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    
    def show_predictions_tab(self, models, selected_model):
        """Display predictions tab content"""
        st.header("🔮 Fraud Detection Predictions")
        
        # Model info
        st.info(f"🎯 Using **{selected_model}** model for predictions")
        
        # Input form
        st.subheader("📝 Enter Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=10000.0, value=100.0, step=10.0)
            time_val = st.number_input("Time (seconds)", min_value=0, value=1000, step=100)
            
            # V1-V5 features (most important)
            v1 = st.slider("V1 Feature", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
            v2 = st.slider("V2 Feature", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
            v3 = st.slider("V3 Feature", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
            v4 = st.slider("V4 Feature", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
            v5 = st.slider("V5 Feature", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        
        with col2:
            # V6-V10 features
            v6 = st.slider("V6 Feature", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
            v7 = st.slider("V7 Feature", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
            v8 = st.slider("V8 Feature", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
            v9 = st.slider("V9 Feature", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
            v10 = st.slider("V10 Feature", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        
        # Create transaction data
        transaction_data = {
            'Time': time_val,
            'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4, 'V5': v5,
            'V6': v6, 'V7': v7, 'V8': v8, 'V9': v9, 'V10': v10,
            'V11': 0.0, 'V12': 0.0, 'V13': 0.0, 'V14': 0.0, 'V15': 0.0,
            'V16': 0.0, 'V17': 0.0, 'V18': 0.0, 'V19': 0.0, 'V20': 0.0,
            'V21': 0.0, 'V22': 0.0, 'V23': 0.0, 'V24': 0.0, 'V25': 0.0,
            'V26': 0.0, 'V27': 0.0, 'V28': 0.0, 'Amount': amount
        }
        
        # Prediction button
        if st.button("🔍 Predict Fraud", type="primary"):
            with st.spinner("Analyzing transaction..."):
                model = models[selected_model]
                prediction, probability = self.predict_fraud(transaction_data, model, selected_model)
                
                if prediction is not None:
                    st.subheader("🎯 Prediction Results")
                    
                    # Display result
                    if prediction == 1:
                        st.markdown('<div class="fraud-alert">🚨 <strong>FRAUD DETECTED!</strong></div>', unsafe_allow_html=True)
                        st.error(f"⚠️ This transaction has been flagged as potentially fraudulent.")
                    else:
                        st.markdown('<div class="legitimate-alert">✅ <strong>LEGITIMATE TRANSACTION</strong></div>', unsafe_allow_html=True)
                        st.success(f"✅ This transaction appears to be legitimate.")
                    
                    # Probability
                    st.metric("Fraud Probability", f"{probability:.3f}")
                    
                    # Confidence level
                    if probability > 0.8:
                        confidence = "Very High"
                        color = "red"
                    elif probability > 0.6:
                        confidence = "High"
                        color = "orange"
                    elif probability > 0.4:
                        confidence = "Medium"
                        color = "yellow"
                    elif probability > 0.2:
                        confidence = "Low"
                        color = "lightgreen"
                    else:
                        confidence = "Very Low"
                        color = "green"
                    
                    st.metric("Confidence Level", confidence)
                    
                    # Transaction summary
                    st.subheader("📋 Transaction Summary")
                    summary_df = pd.DataFrame([transaction_data])
                    st.dataframe(summary_df.T, use_container_width=True)
        
        # Sample transactions
        st.subheader("💡 Try Sample Transactions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💰 High Amount Transaction"):
                sample_data = {
                    'Time': 1000, 'Amount': 5000, 'V1': -5.0, 'V2': 3.0, 'V3': -2.0,
                    'V4': 4.0, 'V5': -1.0, 'V6': 2.0, 'V7': -3.0, 'V8': 1.0, 'V9': -2.0, 'V10': 3.0,
                    'V11': 0.0, 'V12': 0.0, 'V13': 0.0, 'V14': 0.0, 'V15': 0.0,
                    'V16': 0.0, 'V17': 0.0, 'V18': 0.0, 'V19': 0.0, 'V20': 0.0,
                    'V21': 0.0, 'V22': 0.0, 'V23': 0.0, 'V24': 0.0, 'V25': 0.0,
                    'V26': 0.0, 'V27': 0.0, 'V28': 0.0
                }
                st.session_state.sample_transaction = sample_data
                st.success("Sample transaction loaded! Click 'Predict Fraud' to test.")
        
        with col2:
            if st.button("⏰ Late Night Transaction"):
                sample_data = {
                    'Time': 80000, 'Amount': 200, 'V1': 2.0, 'V2': -1.0, 'V3': 3.0,
                    'V4': -2.0, 'V5': 1.0, 'V6': -3.0, 'V7': 2.0, 'V8': -1.0, 'V9': 3.0, 'V10': -2.0,
                    'V11': 0.0, 'V12': 0.0, 'V13': 0.0, 'V14': 0.0, 'V15': 0.0,
                    'V16': 0.0, 'V17': 0.0, 'V18': 0.0, 'V19': 0.0, 'V20': 0.0,
                    'V21': 0.0, 'V22': 0.0, 'V23': 0.0, 'V24': 0.0, 'V25': 0.0,
                    'V26': 0.0, 'V27': 0.0, 'V28': 0.0
                }
                st.session_state.sample_transaction = sample_data
                st.success("Sample transaction loaded! Click 'Predict Fraud' to test.")
        
        with col3:
            if st.button("🔄 Reset Form"):
                st.rerun()
    
    def show_analysis_tab(self):
        """Display analysis tab content"""
        st.header("📊 Data Analysis & Exploration")
        
        # Load data
        df = self.load_data()
        if df is None:
            st.error("❌ Unable to load data for analysis")
            return
        
        # Data overview
        st.subheader("📋 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        
        with col2:
            fraud_count = df['Class'].sum()
            st.metric("Fraudulent Transactions", f"{fraud_count:,}")
        
        with col3:
            fraud_rate = (fraud_count / len(df)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.3f}%")
        
        # Target distribution
        st.subheader("🎯 Transaction Class Distribution")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        df['Class'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        ax.set_title('Transaction Class Distribution')
        ax.set_ylabel('')
        st.pyplot(fig)
        
        # Amount analysis
        st.subheader("💰 Transaction Amount Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Amount distribution by class
            fig, ax = plt.subplots(figsize=(8, 6))
            legitimate_amounts = df[df['Class'] == 0]['Amount']
            fraud_amounts = df[df['Class'] == 1]['Amount']
            
            ax.hist(legitimate_amounts, bins=50, alpha=0.7, label='Legitimate', color='lightblue')
            ax.hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud', color='lightcoral')
            ax.set_xlabel('Transaction Amount')
            ax.set_ylabel('Frequency')
            ax.set_title('Amount Distribution by Class')
            ax.legend()
            ax.set_xlim(0, 1000)
            st.pyplot(fig)
        
        with col2:
            # Amount statistics
            st.subheader("📊 Amount Statistics")
            
            stats_df = df.groupby('Class')['Amount'].agg(['mean', 'std', 'min', 'max']).round(2)
            st.dataframe(stats_df, use_container_width=True)
        
        # Feature correlation
        st.subheader("🔗 Feature Correlation Analysis")
        
        # Select features for correlation
        feature_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'Amount', 'Class']
        corr_data = df[feature_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
        ax.set_title('Feature Correlation Heatmap')
        st.pyplot(fig)
        
        # Feature importance (correlation with target)
        st.subheader("🎯 Feature Importance (Correlation with Fraud)")
        
        feature_importance = pd.DataFrame({
            'Feature': feature_cols[:-1],  # Exclude 'Class'
            'Correlation_with_Class': [abs(corr_data['Class'][col]) for col in feature_cols[:-1]]
        }).sort_values('Correlation_with_Class', ascending=False)
        
        fig = px.bar(feature_importance, 
                    x='Correlation_with_Class', 
                    y='Feature',
                    orientation='h',
                    title='Feature Importance by Correlation with Fraud Class',
                    color='Correlation_with_Class',
                    color_continuous_scale='viridis')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def show_performance_tab(self, results):
        """Display performance tab content"""
        st.header("📈 Model Performance Analysis")
        
        if results is None:
            st.error("❌ No performance results found. Please run the training script first.")
            st.info("💡 Run: `python enhanced_credit_card_fraud_detection.py`")
            return
        
        # Performance overview
        st.subheader("🏆 Overall Performance Summary")
        
        # Best model
        best_model = results['F1-Score'].idxmax()
        best_f1 = results.loc[best_model, 'F1-Score']
        best_auc = results.loc[best_model, 'AUC-ROC']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Model", best_model)
        
        with col2:
            st.metric("Best F1-Score", f"{best_f1:.4f}")
        
        with col3:
            st.metric("Best AUC-ROC", f"{best_auc:.4f}")
        
        # Performance comparison
        st.subheader("📊 Model Performance Comparison")
        
        # Metrics comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, (model_name, color) in enumerate(zip(results.index, ['blue', 'green', 'red'])):
            values = [results.loc[model_name, metric] for metric in metrics]
            ax.bar(x + i*width, values, width, label=model_name, color=color, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Metrics Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Detailed results table
        st.subheader("📋 Detailed Performance Metrics")
        
        # Format results for display
        display_results = results.copy()
        display_results['Training_Time'] = display_results['Training_Time'].round(3)
        
        st.dataframe(display_results, use_container_width=True)
        
        # Performance insights
        st.subheader("💡 Performance Insights")
        
        # Find best model for each metric
        best_accuracy = results['Accuracy'].idxmax()
        best_precision = results['Precision'].idxmax()
        best_recall = results['Recall'].idxmax()
        best_f1 = results['F1-Score'].idxmax()
        best_auc = results['AUC-ROC'].idxmax()
        fastest = results['Training_Time'].idxmin()
        
        insights = f"""
        **🎯 Best Performance by Metric:**
        - **Accuracy**: {best_accuracy} ({results.loc[best_accuracy, 'Accuracy']:.4f})
        - **Precision**: {best_precision} ({results.loc[best_precision, 'Precision']:.4f})
        - **Recall**: {best_recall} ({results.loc[best_recall, 'Recall']:.4f})
        - **F1-Score**: {best_f1} ({results.loc[best_f1, 'F1-Score']:.4f})
        - **AUC-ROC**: {best_auc} ({results.loc[best_auc, 'AUC-ROC']:.4f})
        
        **⚡ Training Speed:**
        - **Fastest**: {fastest} ({results.loc[fastest, 'Training_Time']:.3f}s)
        - **Slowest**: {results['Training_Time'].idxmax()} ({results['Training_Time'].max():.3f}s)
        
        **🏆 Overall Recommendation:**
        Based on F1-Score (balanced measure), **{best_f1}** is the best model for production deployment.
        """
        
        st.markdown(insights)
        
        # Download results
        st.subheader("💾 Download Results")
        
        csv = results.to_csv()
        st.download_button(
            label="📥 Download Performance Results (CSV)",
            data=csv,
            file_name="model_performance_results.csv",
            mime="text/csv"
        )

def main():
    """Main function to run the dashboard"""
    # Initialize dashboard
    dashboard = FraudDetectionDashboard()
    
    # Run dashboard
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
