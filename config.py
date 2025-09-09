# Credit Card Fraud Detection Configuration
# ================================================

# Data Configuration
DATA_PATH = "data/creditcard.csv"
MODELS_DIR = "models"
RESULTS_DIR = "results"
LOGS_DIR = "logs"

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
SMOTE_SAMPLING_STRATEGY = "auto"

# Speed/Showcase Mode
FAST_MODE = True  # Set True to speed up training for demos
FAST_SAMPLE_SIZE = 50000  # Rows to use in fast mode (stratified)
USE_SMOTE = True  # Apply SMOTE to handle imbalance
ENABLE_XGBOOST = False  # XGBoost can be slow; disable for fast demos
CACHE_MODELS = True  # Save models/results
SKIP_TRAIN_IF_CACHED = True  # If models/results exist, skip retraining

# Feature Engineering
FEATURE_ENGINEERING_ENABLED = True
CREATE_INTERACTION_FEATURES = True
CREATE_STATISTICAL_FEATURES = True

# Model Parameters
LOGISTIC_REGRESSION_MAX_ITER = 1000
RANDOM_FOREST_N_ESTIMATORS = 100
XGBOOST_EVAL_METRIC = "logloss"

# Fast-mode parameter overrides
LOGISTIC_REGRESSION_MAX_ITER_FAST = 300
RANDOM_FOREST_N_ESTIMATORS_FAST = 50

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True

# Dashboard Configuration
DASHBOARD_PORT = 8501
DASHBOARD_THEME = "light"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "logs/fraud_detection.log"
