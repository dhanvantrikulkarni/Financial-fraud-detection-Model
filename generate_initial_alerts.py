"""
Generate initial fraud alerts from existing transaction data
This script uses the trained models to score existing transactions and generate alerts
"""
import sys
import os
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_initial_alerts(model_path='models/logistic_regression.pkl', 
                            db_path='fraud_detection.db',
                            threshold=0.8):
    """
    Generate fraud alerts from existing transaction data
    
    Args:
        model_path: Path to trained model
        db_path: Path to SQLite database
        threshold: Fraud probability threshold for alerts
    """
    try:
        logger.info("Starting initial alert generation...")
        
        # Initialize components
        preprocessor = DataPreprocessor()
        
        # Load trained model
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Connect to database and get transactions
        conn = sqlite3.connect(db_path)
        
        # Create fraud_alerts table if it doesn't exist
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fraud_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                amount REAL,
                location TEXT,
                fraud_probability REAL,
                alert_timestamp TEXT,
                model_used TEXT,
                risk_level TEXT,
                notified BOOLEAN DEFAULT 0
            )
        """)
        conn.commit()
        
        # Get all transactions
        query = "SELECT * FROM transactions"
        transactions_df = pd.read_sql(query, conn)
        logger.info(f"Loaded {len(transactions_df)} transactions from database")
        
        if transactions_df.empty:
            logger.warning("No transactions found in database")
            return
        
        # Preprocess transactions
        logger.info("Preprocessing transactions...")
        df_cleaned = preprocessor.handle_missing_values(transactions_df)
        
        # Encode categorical features
        categorical_columns = ['location', 'device', 'merchant_category', 'payment_method']
        categorical_columns = [col for col in categorical_columns if col in df_cleaned.columns]
        if categorical_columns:
            df_cleaned = preprocessor.encode_categorical_features(df_cleaned, categorical_columns)
        
        # Feature engineering
        df_cleaned = preprocessor.feature_engineering(df_cleaned)
        
        # Normalize features
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        exclude_columns = ['transaction_id', 'customer_id', 'is_fraud']
        numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
        if numeric_columns:
            df_cleaned = preprocessor.normalize_features(df_cleaned, numeric_columns)
        
        # Handle any remaining NaN values
        df_cleaned = df_cleaned.fillna(0)
        
        # Prepare features for prediction
        feature_columns = [col for col in df_cleaned.columns 
                         if col not in ['transaction_id', 'customer_id', 'timestamp', 'is_fraud']]
        X = df_cleaned[feature_columns]
        
        # Predict fraud probabilities
        logger.info("Predicting fraud probabilities...")
        if hasattr(model, 'predict_proba'):
            fraud_probabilities = model.predict_proba(X)[:, 1]
        else:
            if hasattr(model, 'decision_function'):
                decisions = model.decision_function(X)
                fraud_probabilities = 1 / (1 + np.exp(-decisions))
            else:
                predictions = model.predict(X)
                fraud_probabilities = predictions.astype(float)
        
        # Generate alerts for high-risk transactions
        logger.info(f"Generating alerts with threshold {threshold}...")
        alerts_generated = 0
        
        for i, (idx, transaction) in enumerate(transactions_df.iterrows()):
            fraud_prob = fraud_probabilities[i]
            
            if fraud_prob >= threshold:
                # Determine risk level
                if fraud_prob > 0.9:
                    risk_level = "HIGH"
                elif fraud_prob > 0.7:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
                
                # Store alert in database
                cursor.execute("""
                    INSERT INTO fraud_alerts 
                    (transaction_id, amount, location, fraud_probability, alert_timestamp, model_used, risk_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    transaction.get('transaction_id', 'N/A'),
                    transaction.get('amount', 0),
                    transaction.get('location', 'N/A'),
                    fraud_prob,
                    datetime.now().isoformat(),
                    model_path.split('/')[-1],
                    risk_level
                ))
                
                alerts_generated += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"Alert generation completed: {alerts_generated} alerts generated from {len(transactions_df)} transactions")
        logger.info(f"Alerts saved to database. View them in the dashboard's '⚠️ Fraud Alerts' page")
        
        return alerts_generated
        
    except Exception as e:
        logger.error(f"Error generating initial alerts: {e}")
        return 0

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate initial fraud alerts from existing data')
    parser.add_argument('--model_path', type=str, default='models/logistic_regression.pkl',
                       help='Path to trained model')
    parser.add_argument('--db_path', type=str, default='fraud_detection.db',
                       help='Path to SQLite database')
    parser.add_argument('--threshold', type=float, default=0.8,
                       help='Fraud probability threshold for alerts')
    
    args = parser.parse_args()
    
    alerts_generated = generate_initial_alerts(
        model_path=args.model_path,
        db_path=args.db_path,
        threshold=args.threshold
    )
    
    print(f"\nGenerated {alerts_generated} fraud alerts")
    print("View alerts in the dashboard at http://localhost:8501")
    print("Navigate to 'Fraud Alerts' page to see the alerts")

if __name__ == "__main__":
    main()