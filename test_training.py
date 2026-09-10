"""
Test script to generate data and train models
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.generate_sample_data import generate_fraud_data
from src.data_preprocessing import DataPreprocessor
from src.supervised_models import SupervisedFraudDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Generating sample data...")
    df = generate_fraud_data(1000, 0.1)
    logger.info(f"Generated data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Initialize components
    preprocessor = DataPreprocessor()
    supervised_detector = SupervisedFraudDetector()
    
    # Preprocess data
    logger.info("Preprocessing data...")
    df = preprocessor.handle_missing_values(df)
    
    # Encode categorical features
    categorical_columns = ['location', 'device', 'merchant_category', 'payment_method']
    categorical_columns = [col for col in categorical_columns if col in df.columns]
    if categorical_columns:
        df = preprocessor.encode_categorical_features(df, categorical_columns)
    
    # Feature engineering
    df = preprocessor.feature_engineering(df)
    
    # Normalize features
    import numpy as np
    all_numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_columns = ['transaction_id', 'customer_id', 'is_fraud']
    numeric_columns = [col for col in all_numeric_columns if col not in exclude_columns]
    
    if numeric_columns:
        df = preprocessor.normalize_features(df, numeric_columns)
    
    # Fill NaN values
    df = df.fillna(0)
    
    # Prepare training data
    logger.info("Preparing training data...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_training_data(df)
    
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Feature columns: {X_train.columns.tolist()}")
    
    # Train a simple model
    logger.info("Training Logistic Regression...")
    supervised_detector.train_logistic_regression(X_train, y_train)
    
    # Evaluate
    logger.info("Evaluating model...")
    results = supervised_detector.evaluate_all_models(X_test, y_test)
    
    logger.info(f"Results: {results}")
    
    # Save model
    supervised_detector.save_models()
    logger.info("Model saved successfully!")

if __name__ == "__main__":
    main()
