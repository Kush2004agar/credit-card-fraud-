
#!/usr/bin/env python3
"""
Credit Card Fraud Detection API
===============================

A FastAPI-based REST API for real-time credit card fraud detection.
Provides endpoints for predictions, model information, and health checks.

Author: [Your Name]
Date: [Current Date]
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import time
from typing import Dict, List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time API for detecting fraudulent credit card transactions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class TransactionData(BaseModel):
    """Transaction data model for fraud detection"""
    Time: float = Field(..., description="Transaction timestamp in seconds")
    V1: float = Field(..., description="Feature V1")
    V2: float = Field(..., description="Feature V2")
    V3: float = Field(..., description="Feature V3")
    V4: float = Field(..., description="Feature V4")
    V5: float = Field(..., description="Feature V5")
    V6: float = Field(..., description="Feature V6")
    V7: float = Field(..., description="Feature V7")
    V8: float = Field(..., description="Feature V8")
    V9: float = Field(..., description="Feature V9")
    V10: float = Field(..., description="Feature V10")
    V11: float = Field(0.0, description="Feature V11")
    V12: float = Field(0.0, description="Feature V12")
    V13: float = Field(0.0, description="Feature V13")
    V14: float = Field(0.0, description="Feature V14")
    V15: float = Field(0.0, description="Feature V15")
    V16: float = Field(0.0, description="Feature V16")
    V17: float = Field(0.0, description="Feature V17")
    V18: float = Field(0.0, description="Feature V18")
    V19: float = Field(0.0, description="Feature V19")
    V20: float = Field(0.0, description="Feature V20")
    V21: float = Field(0.0, description="Feature V21")
    V22: float = Field(0.0, description="Feature V22")
    V23: float = Field(0.0, description="Feature V23")
    V24: float = Field(0.0, description="Feature V24")
    V25: float = Field(0.0, description="Feature V25")
    V26: float = Field(0.0, description="Feature V26")
    V27: float = Field(0.0, description="Feature V27")
    V28: float = Field(0.0, description="Feature V28")
    Amount: float = Field(..., description="Transaction amount in dollars")

class PredictionResponse(BaseModel):
    """Response model for fraud predictions"""
    transaction_id: str
    prediction: int
    probability: float
    confidence: str
    model_used: str
    timestamp: str
    processing_time_ms: float

class ModelInfo(BaseModel):
    """Model information model"""
    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    training_time: float
    last_updated: str

class HealthCheck(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    models_loaded: int
    api_version: str

class FraudDetectionAPI:
    """Main API class for fraud detection"""
    
    def __init__(self):
        self.models_dir = 'models'
        self.results_dir = 'results'
        self.models = {}
        self.results = None
        self.feature_engineering_func = None
        self.load_models()
        self.load_results()
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            if os.path.exists(self.models_dir):
                for file in os.listdir(self.models_dir):
                    if file.endswith('_model.pkl'):
                        model_name = file.replace('_model.pkl', '').replace('_', ' ').title()
                        model_path = os.path.join(self.models_dir, file)
                        self.models[model_name] = joblib.load(model_path)
                        print(f"✅ Loaded {model_name} model")
                
                # Load feature engineering function
                feature_eng_path = os.path.join(self.models_dir, 'feature_engineering.pkl')
                if os.path.exists(feature_eng_path):
                    with open(feature_eng_path, 'rb') as f:
                        self.feature_engineering_func = pickle.load(f)
                    print("✅ Loaded feature engineering function")
            
            print(f"📊 Loaded {len(self.models)} models")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def load_results(self):
        """Load model performance results"""
        try:
            results_path = os.path.join(self.results_dir, 'model_performance.csv')
            if os.path.exists(results_path):
                self.results = pd.read_csv(results_path, index_col=0)
                print("✅ Loaded model performance results")
        except Exception as e:
            print(f"❌ Error loading results: {e}")
    
    def create_features(self, df):
        """Create engineered features"""
        if self.feature_engineering_func:
            return self.feature_engineering_func(df)
        
        # Fallback feature engineering
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
    
    def predict_fraud(self, transaction_data: TransactionData, model_name: str = "XGBoost") -> tuple:
        """Predict fraud for a transaction"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            # Convert to DataFrame
            df_transaction = pd.DataFrame([transaction_data.dict()])
            
            # Apply feature engineering if needed
            if model_name == 'Logistic Regression':
                df_transaction = self.create_features(df_transaction)
            
            # Make prediction
            prediction = model.predict(df_transaction)[0]
            probability = model.predict_proba(df_transaction)[0, 1]
            
            return prediction, probability
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    def get_confidence_level(self, probability: float) -> str:
        """Get confidence level based on probability"""
        if probability > 0.8:
            return "Very High"
        elif probability > 0.6:
            return "High"
        elif probability > 0.4:
            return "Medium"
        elif probability > 0.2:
            return "Low"
        else:
            return "Very Low"

# Initialize API
api = FraudDetectionAPI()

# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        models_loaded=len(api.models),
        api_version="1.0.0"
    )

@app.get("/models", response_model=List[str])
async def get_available_models():
    """Get list of available models"""
    return list(api.models.keys())

@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    if model_name not in api.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    if api.results is not None and model_name in api.results.index:
        model_results = api.results.loc[model_name]
        return ModelInfo(
            name=model_name,
            accuracy=float(model_results['Accuracy']),
            precision=float(model_results['Precision']),
            recall=float(model_results['Recall']),
            f1_score=float(model_results['F1-Score']),
            auc_roc=float(model_results['AUC-ROC']),
            training_time=float(model_results['Training_Time']),
            last_updated=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    else:
        return ModelInfo(
            name=model_name,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            auc_roc=0.0,
            training_time=0.0,
            last_updated=time.strftime("%Y-%m-%d %H:%M:%S")
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionData, model_name: str = "XGBoost"):
    """Predict fraud for a transaction"""
    start_time = time.time()
    
    # Generate transaction ID
    transaction_id = f"txn_{int(time.time())}_{hash(str(transaction.dict())) % 10000}"
    
    # Make prediction
    prediction, probability = api.predict_fraud(transaction, model_name)
    
    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000
    
    # Get confidence level
    confidence = api.get_confidence_level(probability)
    
    return PredictionResponse(
        transaction_id=transaction_id,
        prediction=int(prediction),
        probability=float(probability),
        confidence=confidence,
        model_used=model_name,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        processing_time_ms=round(processing_time, 2)
    )

@app.post("/predict/batch")
async def predict_fraud_batch(transactions: List[TransactionData], model_name: str = "XGBoost"):
    """Predict fraud for multiple transactions"""
    if len(transactions) > 1000:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 1000 transactions")
    
    start_time = time.time()
    results = []
    
    for i, transaction in enumerate(transactions):
        try:
            prediction, probability = api.predict_fraud(transaction, model_name)
            confidence = api.get_confidence_level(probability)
            
            results.append({
                "transaction_index": i,
                "prediction": int(prediction),
                "probability": float(probability),
                "confidence": confidence,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "transaction_index": i,
                "prediction": None,
                "probability": None,
                "confidence": None,
                "status": "error",
                "error": str(e)
            })
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "batch_id": f"batch_{int(time.time())}",
        "total_transactions": len(transactions),
        "successful_predictions": len([r for r in results if r['status'] == 'success']),
        "failed_predictions": len([r for r in results if r['status'] == 'error']),
        "processing_time_ms": round(processing_time, 2),
        "model_used": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results
    }

@app.get("/performance", response_model=Dict)
async def get_performance_summary():
    """Get overall performance summary"""
    if api.results is None:
        raise HTTPException(status_code=404, detail="Performance results not found")
    
    # Calculate summary statistics
    summary = {
        "total_models": len(api.results),
        "best_model_f1": api.results['F1-Score'].idxmax(),
        "best_f1_score": float(api.results['F1-Score'].max()),
        "best_model_auc": api.results['AUC-ROC'].idxmax(),
        "best_auc_score": float(api.results['AUC-ROC'].max()),
        "fastest_model": api.results['Training_Time'].idxmin(),
        "fastest_training_time": float(api.results['Training_Time'].min()),
        "average_accuracy": float(api.results['Accuracy'].mean()),
        "average_f1_score": float(api.results['F1-Score'].mean()),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return summary

@app.get("/performance/{model_name}")
async def get_model_performance(model_name: str):
    """Get performance metrics for a specific model"""
    if api.results is None:
        raise HTTPException(status_code=404, detail="Performance results not found")
    
    if model_name not in api.results.index:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model_results = api.results.loc[model_name]
    
    return {
        "model_name": model_name,
        "accuracy": float(model_results['Accuracy']),
        "precision": float(model_results['Precision']),
        "recall": float(model_results['Recall']),
        "f1_score": float(model_results['F1-Score']),
        "auc_roc": float(model_results['AUC-ROC']),
        "training_time": float(model_results['Training_Time']),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/reload-models")
async def reload_models(background_tasks: BackgroundTasks):
    """Reload models from disk (background task)"""
    def reload():
        time.sleep(1)  # Simulate reload time
        api.load_models()
        api.load_results()
    
    background_tasks.add_task(reload)
    
    return {
        "message": "Model reload initiated",
        "status": "reloading",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize API on startup"""
    print("🚀 Starting Credit Card Fraud Detection API...")
    print(f"📊 Loaded {len(api.models)} models")
    if api.results is not None:
        print(f"📈 Loaded performance results for {len(api.results)} models")
    print("✅ API ready to serve requests!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down Credit Card Fraud Detection API...")

if __name__ == "__main__":
    # Run the API
    print("🔧 Starting Credit Card Fraud Detection API...")
    print("📚 API documentation available at: http://localhost:8000/docs")
    print("🔍 Health check available at: http://localhost:8000/health")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
