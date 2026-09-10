"""
Main training script for Financial Fraud Detection Model
"""
import sys
import os
import argparse
import logging
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import DataPreprocessor
from src.supervised_models import SupervisedFraudDetector
from src.unsupervised_models import UnsupervisedFraudDetector
from src.generate_sample_data import generate_fraud_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train Fraud Detection Models')
    parser.add_argument('--data_path', type=str, default='data/financial_fraud_detection_dataset.csv',
                       help='Path to training data file')
    parser.add_argument('--generate_data', action='store_true',
                       help='Generate sample data instead of loading from file')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--fraud_ratio', type=float, default=0.1,
                       help='Ratio of fraudulent transactions')
    parser.add_argument('--model_type', type=str, default='all',
                       choices=['supervised', 'unsupervised', 'all'],
                       help='Type of models to train')
    parser.add_argument('--supervised_models', nargs='+', 
                       choices=['logistic_regression', 'random_forest', 'xgboost', 'lightgbm'],
                       default=['logistic_regression', 'random_forest'],
                       help='Supervised models to train')
    parser.add_argument('--unsupervised_models', nargs='+',
                       choices=['isolation_forest', 'one_class_svm', 'autoencoder'],
                       default=['isolation_forest'],
                       help='Unsupervised models to train')
    
    args = parser.parse_args()
    
    logger.info("Starting Fraud Detection Model Training")
    
    # Initialize components
    preprocessor = DataPreprocessor()
    supervised_detector = SupervisedFraudDetector()
    unsupervised_detector = UnsupervisedFraudDetector()
    
    # Load or generate data
    if args.generate_data:
        logger.info(f"Generating {args.num_samples} samples with fraud ratio {args.fraud_ratio}")
        df = generate_fraud_data(args.num_samples, args.fraud_ratio)
    else:
        logger.info(f"Loading data from {args.data_path}")
        df = preprocessor.load_data(args.data_path)
    
    # Normalize column names if loading from database with original names
    if not args.generate_data:
        column_mapping = {
            'Transaction_ID': 'transaction_id',
            'Customer_ID': 'customer_id', 
            'Transaction_Date': 'timestamp',
            'Transaction_Amount': 'amount',
            'Merchant_Category': 'merchant_category',
            'Payment_Method': 'payment_method',
            'Device_Type': 'device',
            'Location': 'location',
            'Is_International': 'is_international',
            'Previous_Transactions': 'previous_transactions',
            'Average_Spend': 'average_spend',
            'Account_Age_Days': 'account_age_days',
            'Suspicious_Keyword': 'suspicious_keyword',
            'Fraudulent': 'is_fraud'
        }
        
        # Apply column mapping if columns exist
        mapping_to_apply = {}
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                mapping_to_apply[old_col] = new_col
        
        if mapping_to_apply:
            df = df.rename(columns=mapping_to_apply)
            logger.info(f"Normalized column names: {mapping_to_apply}")
    
    # Preprocess data
    logger.info("Preprocessing data...")
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df)
    
    # Encode categorical features
    categorical_columns = ['location', 'device', 'merchant_category', 'payment_method']
    categorical_columns = [col for col in categorical_columns if col in df.columns]
    if categorical_columns:
        df = preprocessor.encode_categorical_features(df, categorical_columns)
    
    # Feature engineering
    df = preprocessor.feature_engineering(df)
    
    # Normalize features - use only numeric columns that exist
    all_numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove ID columns and target from numeric columns
    exclude_columns = ['transaction_id', 'customer_id', 'is_fraud']
    numeric_columns = [col for col in all_numeric_columns if col not in exclude_columns]
    
    if numeric_columns:
        df = preprocessor.normalize_features(df, numeric_columns)
    
    # Additional NaN handling - ensure no NaN values remain
    df = df.fillna(0)
    if df.isnull().any().any():
        logger.warning("NaN values still present after initial fill, filling again")
        df = df.fillna(0)
    
    # Prepare training data
    X_train, X_test, y_train, y_test = preprocessor.prepare_training_data(df)
    
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train models
    if args.model_type in ['supervised', 'all']:
        logger.info("Training supervised models...")
        
        if 'logistic_regression' in args.supervised_models:
            supervised_detector.train_logistic_regression(X_train, y_train)
        
        if 'random_forest' in args.supervised_models:
            supervised_detector.train_random_forest(X_train, y_train)
        
        if 'xgboost' in args.supervised_models:
            supervised_detector.train_xgboost(X_train, y_train)
        
        if 'lightgbm' in args.supervised_models:
            supervised_detector.train_lightgbm(X_train, y_train)
        
        # Evaluate supervised models
        supervised_results = supervised_detector.evaluate_all_models(X_test, y_test)
        
        # Print results
        logger.info("\n=== Supervised Model Results ===")
        for model_name, results in supervised_results.items():
            logger.info(f"{model_name}:")
            logger.info(f"  AUC Score: {results['auc_score']:.4f}")
            logger.info(f"  Precision: {results['classification_report']['weighted avg']['precision']:.4f}")
            logger.info(f"  Recall: {results['classification_report']['weighted avg']['recall']:.4f}")
            logger.info(f"  F1-Score: {results['classification_report']['weighted avg']['f1-score']:.4f}")
        
        # Save models and results
        supervised_detector.save_models()
        import joblib
        joblib.dump(supervised_results, 'models/supervised_results.pkl')
        logger.info("Supervised models saved successfully")
    
    if args.model_type in ['unsupervised', 'all']:
        logger.info("Training unsupervised models...")
        
        if 'isolation_forest' in args.unsupervised_models:
            unsupervised_detector.train_isolation_forest(X_train)
        
        if 'one_class_svm' in args.unsupervised_models:
            unsupervised_detector.train_one_class_svm(X_train)
        
        if 'autoencoder' in args.unsupervised_models:
            unsupervised_detector.train_autoencoder(X_train)
        
        # Evaluate unsupervised models
        unsupervised_results = unsupervised_detector.evaluate_all_models(X_test, y_test)
        
        # Print results
        logger.info("\n=== Unsupervised Model Results ===")
        for model_name, results in unsupervised_results.items():
            logger.info(f"{model_name}:")
            logger.info(f"  AUC Score: {results['auc_score']:.4f}")
            logger.info(f"  Precision: {results['classification_report']['weighted avg']['precision']:.4f}")
            logger.info(f"  Recall: {results['classification_report']['weighted avg']['recall']:.4f}")
            logger.info(f"  F1-Score: {results['classification_report']['weighted avg']['f1-score']:.4f}")
        
        # Save models and results
        unsupervised_detector.save_models()
        import joblib
        joblib.dump(unsupervised_results, 'models/unsupervised_results.pkl')
        logger.info("Unsupervised models saved successfully")
    
    # ETL process to save to database
    logger.info("Running ETL process to save data to database...")
    preprocessor.etl_process(df)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
