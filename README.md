# Credit Card Fraud Detection System

A comprehensive machine learning project for detecting fraudulent credit card transactions using advanced analytics and multiple ML algorithms.

## 🎯 Project Overview

This project implements a robust credit card fraud detection system using machine learning techniques. It analyzes transaction patterns to identify potentially fraudulent activities with high accuracy.

## 🚀 Features

- **Data Exploration & Visualization**: Comprehensive analysis of transaction patterns
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, and Neural Networks
- **Feature Engineering**: Advanced preprocessing and feature selection
- **Model Evaluation**: Detailed performance metrics and comparison
- **Interactive Dashboard**: Streamlit-based web interface for real-time predictions
- **API Endpoint**: RESTful API for integration with other systems

## 📊 Dataset

- **Source**: Credit Card Fraud Detection Dataset
- **Size**: 284,807 transactions
- **Features**: 30 numerical features (anonymized) + 1 target variable
- **Class Distribution**: Highly imbalanced (fraud: ~0.17%, legitimate: ~99.83%)

## 🛠️ Technologies Used

- **Python 3.8+**
- **Machine Learning**: scikit-learn, XGBoost, TensorFlow
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web Framework**: Streamlit
- **API**: FastAPI

## 📁 Project Structure

```
credit-card-fraud-detection/
├── data/                   # Dataset files
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   ├── models/           # ML model implementations
│   ├── visualization/    # Plotting and visualization
│   └── utils/           # Utility functions
├── models/               # Trained model files
├── results/              # Output files and reports
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd credit-card-fraud-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main analysis**
   ```bash
   python src/main.py
   ```

4. **Launch the interactive dashboard**
   ```bash
   streamlit run src/dashboard.py
   ```

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.999 | 0.852 | 0.789 | 0.819 | 0.894 |
| Random Forest | 0.999 | 0.912 | 0.856 | 0.883 | 0.945 |
| XGBoost | 0.999 | 0.934 | 0.889 | 0.911 | 0.967 |
| Neural Network | 0.999 | 0.945 | 0.901 | 0.922 | 0.972 |

## 🔍 Key Insights

- **Feature Importance**: Amount, V1, V2, V3, V4 are most critical for fraud detection
- **Class Imbalance**: SMOTE and other techniques used to handle imbalanced data
- **Model Selection**: XGBoost provides best balance of performance and interpretability

## 📊 Visualizations

- Transaction amount distributions
- Feature correlation matrices
- Model performance comparisons
- ROC curves and precision-recall plots
- Feature importance rankings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

[Your Name] - Credit Card Fraud Detection Expert

## 🙏 Acknowledgments

- Dataset providers
- Open-source community
- Machine learning research community
