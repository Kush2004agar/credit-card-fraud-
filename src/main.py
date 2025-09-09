#!/usr/bin/env python3
"""
Enhanced Credit Card Fraud Detection System
==========================================

A comprehensive machine learning project for detecting fraudulent credit card 
transactions with advanced analytics, multiple ML models, and detailed evaluation.

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os
import joblib
import pickle
import config as cfg

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score
)

# Advanced ML
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("⚠️  SMOTE not available. Install with: pip install imbalanced-learn")
    SMOTE_AVAILABLE = False

# Set style
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class CreditCardFraudDetector:
    """
    Comprehensive credit card fraud detection system
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the fraud detector
        
        Parameters:
        data_path (str): Path to the credit card dataset
        """
        self.data_path = data_path or cfg.DATA_PATH
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("📊 Loading credit card fraud dataset...")
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"📈 Shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")
            
            # Basic information
            print(f"\n📊 Dataset Information:")
            print("=" * 50)
            print(f"Total transactions: {len(self.df):,}")
            print(f"Total features: {self.df.shape[1] - 1}")
            print(f"Target variable: {self.df.columns[-1]}")
            
            # Check for missing values
            missing_values = self.df.isnull().sum().sum()
            print(f"Missing values: {missing_values}")
            
            # Target distribution
            target_counts = self.df['Class'].value_counts()
            print(f"\n🎯 Target distribution:")
            print(target_counts)
            print(f"Fraud percentage: {(target_counts[1] / len(self.df)) * 100:.4f}%")
            print(f"Legitimate percentage: {(target_counts[0] / len(self.df)) * 100:.4f}%")
            
            # Fast/demo mode sampling (stratified)
            if getattr(cfg, 'FAST_MODE', False) and len(self.df) > getattr(cfg, 'FAST_SAMPLE_SIZE', 50000):
                print(f"\n⚡ FAST_MODE enabled: sampling {cfg.FAST_SAMPLE_SIZE:,} rows stratified by class for quicker training")
                total_rows = cfg.FAST_SAMPLE_SIZE
                class_counts = self.df['Class'].value_counts(normalize=True)
                # Ensure at least 1 row per class when possible
                rows_per_class = {
                    c: max(1, int(round(total_rows * frac))) for c, frac in class_counts.items()
                }
                # Adjust to exact total due to rounding
                diff = total_rows - sum(rows_per_class.values())
                if diff != 0:
                    # Assign remainder to the majority class
                    majority_class = class_counts.idxmax()
                    rows_per_class[majority_class] = max(1, rows_per_class.get(majority_class, 0) + diff)
                sampled_parts = []
                rng = cfg.RANDOM_STATE
                for cls, n_rows in rows_per_class.items():
                    group = self.df[self.df['Class'] == cls]
                    n_rows = min(len(group), n_rows)
                    sampled_parts.append(group.sample(n=n_rows, random_state=rng))
                self.df = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=cfg.RANDOM_STATE).reset_index(drop=True)
                print(f"✅ Sampled shape: {self.df.shape}")
                print(self.df['Class'].value_counts())
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def prepare_data(self, test_size=None, random_state=None):
        """Prepare data for training"""
        print("\n🔧 Preparing data for training...")
        if test_size is None:
            test_size = cfg.TEST_SIZE
        if random_state is None:
            random_state = cfg.RANDOM_STATE
        
        # Separate features and target
        self.X = self.df.drop('Class', axis=1)
        self.y = self.df['Class']
        
        print(f"📊 Features shape: {self.X.shape}")
        print(f"🎯 Target shape: {self.y.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"\n📈 Training set: {self.X_train.shape[0]:,} samples")
        print(f"📊 Test set: {self.X_test.shape[0]:,} samples")
        print(f"\n🎯 Training target distribution:")
        print(self.y_train.value_counts())
        print(f"\n🎯 Test target distribution:")
        print(self.y_test.value_counts())
        
        # Handle class imbalance
        if getattr(cfg, 'USE_SMOTE', True) and SMOTE_AVAILABLE:
            print("\n🔄 Applying SMOTE to handle class imbalance...")
            smote = SMOTE(random_state=random_state, sampling_strategy='auto')
            self.X_train_balanced, self.y_train_balanced = smote.fit_resample(self.X_train, self.y_train)
            
            print(f"📊 Before SMOTE - Training set: {self.X_train.shape[0]:,} samples")
            print(f"📊 After SMOTE - Training set: {self.X_train_balanced.shape[0]:,} samples")
            print(f"\n🎯 After SMOTE - Target distribution:")
            print(self.y_train_balanced.value_counts())
        else:
            if not SMOTE_AVAILABLE:
                print("⚠️  SMOTE not available. Using original training data.")
            else:
                print("ℹ️  SMOTE disabled by configuration. Using original training data.")
            self.X_train_balanced, self.y_train_balanced = self.X_train, self.y_train
    
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
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("\n🤖 Training machine learning models...")
        
        # Initialize models
        max_iter = cfg.LOGISTIC_REGRESSION_MAX_ITER
        n_estimators = cfg.RANDOM_FOREST_N_ESTIMATORS
        if getattr(cfg, 'FAST_MODE', False):
            max_iter = getattr(cfg, 'LOGISTIC_REGRESSION_MAX_ITER_FAST', max_iter)
            n_estimators = getattr(cfg, 'RANDOM_FOREST_N_ESTIMATORS_FAST', n_estimators)

        self.models = {
            'Logistic Regression': LogisticRegression(random_state=cfg.RANDOM_STATE, max_iter=max_iter),
            'Random Forest': RandomForestClassifier(random_state=cfg.RANDOM_STATE, n_estimators=n_estimators),
        }
        
        if getattr(cfg, 'ENABLE_XGBOOST', True) and XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(random_state=cfg.RANDOM_STATE, eval_metric=cfg.XGBOOST_EVAL_METRIC)
        
        # Train and evaluate models
        for name, model in self.models.items():
            print(f"\n🚀 Training {name}...")
            start_time = time.time()
            
            # Train the model
            if name == 'Logistic Regression':
                # Use engineered features for logistic regression
                X_train_eng = self.create_features(self.X_train_balanced)
                X_test_eng = self.create_features(self.X_test)
                model.fit(X_train_eng, self.y_train_balanced)
                y_pred = model.predict(X_test_eng)
                y_pred_proba = model.predict_proba(X_test_eng)[:, 1]
            else:
                # Use original features for tree-based models
                model.fit(self.X_train_balanced, self.y_train_balanced)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            training_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            auc_roc = roc_auc_score(self.y_test, y_pred_proba)
            
            self.results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'AUC-ROC': auc_roc,
                'Training_Time': training_time
            }
            
            print(f"✅ {name} trained in {training_time:.2f} seconds")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   AUC-ROC: {auc_roc:.4f}")
        
        # Find best model
        self.best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['F1-Score'])
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 Best Model (by F1-Score): {self.best_model_name}")
        print(f"   F1-Score: {self.results[self.best_model_name]['F1-Score']:.4f}")
        print(f"   AUC-ROC: {self.results[self.best_model_name]['AUC-ROC']:.4f}")
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n📊 Model Performance Comparison")
        print("=" * 60)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        
        # Display results table
        print(results_df)
        
        # Create visualizations
        self.create_performance_plots()
        
        # Detailed analysis of best model
        self.analyze_best_model()
    
    def create_performance_plots(self):
        """Create performance comparison plots"""
        print("\n📈 Creating performance visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Metrics comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, (model_name, color) in enumerate(zip(self.results.keys(), ['blue', 'green', 'red'])):
            values = [self.results[model_name][metric] for metric in metrics]
            axes[0, 0].bar(x + i*width, values, width, label=model_name, color=color, alpha=0.8)
        
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Metrics')
        axes[0, 0].set_xticks(x + width)
        axes[0, 0].set_xticklabels(metrics)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)
        
        # 2. AUC-ROC comparison
        auc_scores = [self.results[model]['AUC-ROC'] for model in self.results.keys()]
        axes[0, 1].bar(self.results.keys(), auc_scores, color=['blue', 'green', 'red'][:len(self.results)], alpha=0.8)
        axes[0, 1].set_ylabel('AUC-ROC Score')
        axes[0, 1].set_title('AUC-ROC Comparison')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(auc_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. Training time comparison
        training_times = [self.results[model]['Training_Time'] for model in self.results.keys()]
        axes[1, 0].bar(self.results.keys(), training_times, color=['blue', 'green', 'red'][:len(self.results)], alpha=0.8)
        axes[1, 0].set_ylabel('Training Time (seconds)')
        axes[1, 0].set_title('Training Time Comparison')
        for i, v in enumerate(training_times):
            axes[1, 0].text(i, v + 0.1, f'{v:.2f}s', ha='center', va='bottom')
        
        # 4. ROC curves
        for name, model in self.models.items():
            if name == 'Logistic Regression':
                X_test_eng = self.create_features(self.X_test)
                y_pred_proba = model.predict_proba(X_test_eng)[:, 1]
            else:
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            axes[1, 1].plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('ROC Curves')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_best_model(self):
        """Detailed analysis of the best performing model"""
        print(f"\n🔍 Detailed Analysis of Best Model: {self.best_model_name}")
        print("=" * 60)
        
        # Get predictions
        if self.best_model_name == 'Logistic Regression':
            X_test_eng = self.create_features(self.X_test)
            y_pred_best = self.best_model.predict(X_test_eng)
            y_pred_proba_best = self.best_model.predict_proba(X_test_eng)[:, 1]
        else:
            y_pred_best = self.best_model.predict(self.X_test)
            y_pred_proba_best = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred_best)
        print(f"\n📊 Confusion Matrix:")
        print(cm)
        
        # Detailed classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(self.y_test, y_pred_best))
        
        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            print(f"\n🔍 Feature Importance (Top 10):")
            if self.best_model_name == 'Logistic Regression':
                feature_names = X_test_eng.columns
                importances = np.abs(self.best_model.coef_[0])
            else:
                feature_names = self.X_test.columns
                importances = self.best_model.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print(feature_importance_df.head(10))
    
    def save_models(self):
        """Save trained models and results"""
        print("\n💾 Saving models and results...")
        
        # Create directories
        os.makedirs(cfg.MODELS_DIR, exist_ok=True)
        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
        
        # Save all models
        for name, model in self.models.items():
            model_path = os.path.join(cfg.MODELS_DIR, f'{name.lower().replace(" ", "_")}_model.pkl')
            joblib.dump(model, model_path)
            print(f"✅ {name} saved to: {model_path}")
        
        # Save results
        results_df = pd.DataFrame(self.results).T
        results_path = os.path.join(cfg.RESULTS_DIR, 'model_performance.csv')
        results_df.to_csv(results_path)
        print(f"✅ Results saved to: {results_path}")
        
        # Save feature engineering function
        with open(os.path.join(cfg.MODELS_DIR, 'feature_engineering.pkl'), 'wb') as f:
            pickle.dump(self.create_features, f)
        print(f"✅ Feature engineering function saved to: {os.path.join(cfg.MODELS_DIR, 'feature_engineering.pkl')}")
        
        print(f"\n🎉 All models and results saved successfully!")
    
    def create_prediction_function(self):
        """Create a prediction function for new transactions"""
        print("\n🔮 Creating prediction function...")
        
        prediction_code = '''
def predict_fraud(transaction_data, model_name='XGBoost'):
    """
    Predict fraud for a single transaction
    
    Parameters:
    transaction_data: dict or pandas Series with transaction features
    model_name: name of the model to use
    
    Returns:
    prediction: 0 (legitimate) or 1 (fraud)
    probability: probability of fraud
    """
    
    # Load the model
    model_path = f'models/{model_name.lower().replace(" ", "_")}_model.pkl'
    model = joblib.load(model_path)
    
    # Convert to DataFrame
    if isinstance(transaction_data, dict):
        df_transaction = pd.DataFrame([transaction_data])
    else:
        df_transaction = pd.DataFrame([transaction_data])
    
    # Apply feature engineering if needed
    if model_name == 'Logistic Regression':
        with open('models/feature_engineering.pkl', 'rb') as f:
            create_features_func = pickle.load(f)
        df_transaction = create_features_func(df_transaction)
    
    # Make prediction
    prediction = model.predict(df_transaction)[0]
    probability = model.predict_proba(df_transaction)[0, 1]
    
    return prediction, probability

# Example usage:
# prediction, prob = predict_fraud(transaction_dict, 'XGBoost')
# print(f'Fraud: {prediction}, Probability: {prob:.3f}')
'''
        
        # Save prediction function
        with open(os.path.join(cfg.MODELS_DIR, 'prediction_function.py'), 'w') as f:
            f.write(prediction_code)
        
        print(f"✅ Prediction function saved to: {os.path.join(cfg.MODELS_DIR, 'prediction_function.py')}")
        print(f"📚 Check the file for usage examples.")
    
    def run_complete_analysis(self):
        """Run the complete fraud detection analysis"""
        print("🚀 Starting Enhanced Credit Card Fraud Detection Analysis")
        print("=" * 70)
        
        # Load data
        if not self.load_data():
            return False
        
        # Prepare data
        self.prepare_data()
        
        # Skip training if cached and configured
        cached_results_path = os.path.join(cfg.RESULTS_DIR, 'model_performance.csv')
        models_exist = os.path.isdir(cfg.MODELS_DIR) and any(f.endswith('_model.pkl') for f in os.listdir(cfg.MODELS_DIR))
        if getattr(cfg, 'SKIP_TRAIN_IF_CACHED', False) and os.path.exists(cached_results_path) and models_exist:
            print("\n🗂️  Cached models and results found. Skipping training as per configuration.")
            try:
                # Load models
                self.models = {}
                for file in os.listdir(cfg.MODELS_DIR):
                    if file.endswith('_model.pkl'):
                        model_name = file.replace('_model.pkl', '').replace('_', ' ').title()
                        self.models[model_name] = joblib.load(os.path.join(cfg.MODELS_DIR, file))
                # Load results
                self.results = pd.read_csv(cached_results_path, index_col=0).to_dict(orient='index')
                # Determine best model
                self.best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['F1-Score'])
                self.best_model = self.models.get(self.best_model_name)
            except Exception as e:
                print(f"⚠️  Failed to load cache ({e}). Proceeding to train models.")
                self.train_models()
        else:
            # Train models
            self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Save models
        if getattr(cfg, 'CACHE_MODELS', True):
            self.save_models()
        
        # Create prediction function
        self.create_prediction_function()
        
        # Summary
        self.print_summary()
        
        return True
    
    def print_summary(self):
        """Print summary of key findings"""
        print("\n🎯 Key Insights and Findings")
        print("=" * 50)
        
        print(f"\n🏆 Best Performing Model: {self.best_model_name}")
        print(f"   - F1-Score: {self.results[self.best_model_name]['F1-Score']:.4f}")
        print(f"   - AUC-ROC: {self.results[self.best_model_name]['AUC-ROC']:.4f}")
        print(f"   - Training Time: {self.results[self.best_model_name]['Training_Time']:.2f} seconds")
        
        print(f"\n📊 Dataset Characteristics:")
        print(f"   - Total Transactions: {len(self.df):,}")
        target_counts = self.df['Class'].value_counts()
        print(f"   - Fraud Rate: {(target_counts[1] / len(self.df)) * 100:.4f}%")
        print(f"   - Features: {self.X.shape[1]}")
        
        print(f"\n🔍 Model Performance Summary:")
        for model_name, metrics in self.results.items():
            print(f"   {model_name}:")
            print(f"     - F1-Score: {metrics['F1-Score']:.4f}")
            print(f"     - AUC-ROC: {metrics['AUC-ROC']:.4f}")
            print(f"     - Training Time: {metrics['Training_Time']:.2f}s")
        
        print(f"\n💡 Recommendations:")
        print(f"   1. Use {self.best_model_name} for production deployment")
        print(f"   2. Monitor model performance regularly")
        print(f"   3. Consider ensemble methods for improved robustness")
        print(f"   4. Implement real-time fraud detection system")
        
        print(f"\n🎉 Project Summary:")
        print(f"   ✅ Comprehensive data analysis completed")
        print(f"   ✅ Multiple ML models trained and evaluated")
        print(f"   ✅ Best model identified and saved")
        print(f"   ✅ Production-ready prediction function created")
        print(f"   ✅ Professional documentation and structure")


def main():
    """Main function to run the fraud detection system"""
    # Initialize the fraud detector
    detector = CreditCardFraudDetector()
    
    # Run complete analysis
    success = detector.run_complete_analysis()
    
    if success:
        print("\n🎯 Your credit card fraud detection project is now showcase-worthy!")
        print("📚 Check the 'models/' and 'results/' directories for saved files.")
        print("🚀 Ready for deployment and presentation!")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
