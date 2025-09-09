#!/usr/bin/env python3
"""
Credit Card Fraud Detection Showcase
====================================

This script demonstrates all the features of the fraud detection system
for showcase and presentation purposes.

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

def print_header():
    """Print showcase header"""
    print("=" * 80)
    print("🎭 Credit Card Fraud Detection System - Showcase")
    print("=" * 80)
    print()

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if dataset exists
    if not os.path.exists('data/creditcard.csv'):
        print("❌ Dataset not found. Please ensure 'data/creditcard.csv' exists.")
        return False
    print("✅ Dataset found")
    
    # Check if models exist
    models_dir = Path('models')
    if not models_dir.exists() or not list(models_dir.glob('*.pkl')):
        print("⚠️  No trained models found. Will need to train models first.")
    else:
        print("✅ Trained models found")
    
    print()
    return True

def run_analysis():
    """Run the main analysis"""
    print("🚀 Running Credit Card Fraud Detection Analysis...")
    print("This will train models and generate performance metrics.")
    print()
    
    try:
        # Run the main analysis script
        result = subprocess.run([sys.executable, 'src/main.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Analysis completed successfully!")
            print("📊 Check the 'results/' directory for performance metrics.")
        else:
            print("❌ Analysis failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Analysis timed out (5 minutes). This is normal for large datasets.")
        return False
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return False
    
    print()
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🌐 Launching Interactive Dashboard...")
    print("The dashboard will open in your default web browser.")
    print()
    
    try:
        # Start dashboard in background
        dashboard_process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 'src/dashboard.py',
            '--server.port', '8501',
            '--server.headless', 'true'
        ])
        
        # Wait a bit for dashboard to start
        time.sleep(5)
        
        # Open dashboard in browser
        webbrowser.open('http://localhost:8501')
        
        print("✅ Dashboard launched successfully!")
        print("🌐 URL: http://localhost:8501")
        print("💡 Press Ctrl+C to stop the dashboard when done.")
        
        # Wait for user to stop
        try:
            dashboard_process.wait()
        except KeyboardInterrupt:
            dashboard_process.terminate()
            print("\n🛑 Dashboard stopped.")
        
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    print()
    return True

def launch_api():
    """Launch the FastAPI server"""
    print("🔌 Launching REST API...")
    print("The API will be available for testing and integration.")
    print()
    
    try:
        # Start API in background
        api_process = subprocess.Popen([
            sys.executable, 'src/api.py'
        ])
        
        # Wait a bit for API to start
        time.sleep(3)
        
        print("✅ API launched successfully!")
        print("🔌 Base URL: http://localhost:8000")
        print("📚 Documentation: http://localhost:8000/docs")
        print("🔍 Health Check: http://localhost:8000/health")
        print("💡 Press Ctrl+C to stop the API when done.")
        
        # Wait for user to stop
        try:
            api_process.wait()
        except KeyboardInterrupt:
            api_process.terminate()
            print("\n🛑 API stopped.")
        
    except Exception as e:
        print(f"❌ Error launching API: {e}")
        return False
    
    print()
    return True

def demonstrate_features():
    """Demonstrate key features"""
    print("🎯 Demonstrating Key Features...")
    print()
    
    # Feature 1: Data Analysis
    print("📊 1. Data Analysis & Visualization")
    print("   - Comprehensive dataset exploration")
    print("   - Feature correlation analysis")
    print("   - Transaction pattern visualization")
    print()
    
    # Feature 2: Machine Learning
    print("🤖 2. Machine Learning Models")
    print("   - Multiple algorithms (Logistic Regression, Random Forest, XGBoost)")
    print("   - Advanced feature engineering")
    print("   - Class imbalance handling with SMOTE")
    print()
    
    # Feature 3: Model Evaluation
    print("📈 3. Model Evaluation & Comparison")
    print("   - Performance metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC)")
    print("   - ROC curves and confusion matrices")
    print("   - Training time comparison")
    print()
    
    # Feature 4: Real-time Predictions
    print("🔮 4. Real-time Fraud Detection")
    print("   - Instant transaction analysis")
    print("   - Probability scores and confidence levels")
    print("   - Batch processing capabilities")
    print()
    
    # Feature 5: Interactive Dashboard
    print("🌐 5. Interactive Web Dashboard")
    print("   - Real-time predictions")
    print("   - Data exploration tools")
    print("   - Model performance visualization")
    print()
    
    # Feature 6: REST API
    print("🔌 6. Production-Ready API")
    print("   - RESTful endpoints")
    print("   - JSON request/response")
    print("   - Comprehensive documentation")
    print()
    
    # Feature 7: Deployment Ready
    print("🚀 7. Deployment & Scalability")
    print("   - Docker containerization")
    print("   - Cloud deployment support")
    print("   - Monitoring and health checks")
    print()

def show_performance_metrics():
    """Show performance metrics if available"""
    print("📊 Performance Metrics (if models are trained)...")
    print()
    
    results_file = Path('results/model_performance.csv')
    if results_file.exists():
        try:
            import pandas as pd
            results = pd.read_csv(results_file, index_col=0)
            
            print("🏆 Model Performance Summary:")
            print("=" * 50)
            print(results.round(4))
            print()
            
            # Find best model
            best_model = results['F1-Score'].idxmax()
            best_f1 = results.loc[best_model, 'F1-Score']
            best_auc = results.loc[best_model, 'AUC-ROC']
            
            print(f"🎯 Best Model: {best_model}")
            print(f"   F1-Score: {best_f1:.4f}")
            print(f"   AUC-ROC: {best_auc:.4f}")
            print()
            
        except Exception as e:
            print(f"⚠️  Could not read performance results: {e}")
    else:
        print("⚠️  No performance results found. Run the analysis first.")
    
    print()

def main():
    """Main showcase function"""
    print_header()
    
    # Check requirements
    if not check_requirements():
        print("❌ System requirements not met. Please fix the issues above.")
        return
    
    # Show features
    demonstrate_features()
    
    # Show performance if available
    show_performance_metrics()
    
    # Menu
    while True:
        print("🎭 Showcase Menu:")
        print("1. 🚀 Run Complete Analysis (Train Models)")
        print("2. 🌐 Launch Interactive Dashboard")
        print("3. 🔌 Launch REST API")
        print("4. 📊 View Performance Metrics")
        print("5. 🎯 Exit Showcase")
        print()
        
        choice = input("Enter your choice (1-5): ").strip()
        print()
        
        if choice == '1':
            run_analysis()
        elif choice == '2':
            launch_dashboard()
        elif choice == '3':
            launch_api()
        elif choice == '4':
            show_performance_metrics()
        elif choice == '5':
            print("🎉 Thank you for exploring the Credit Card Fraud Detection System!")
            print("🚀 Your project is now showcase-ready!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")
        
        print()

if __name__ == "__main__":
    main()
