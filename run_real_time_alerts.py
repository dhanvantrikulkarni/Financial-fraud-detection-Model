"""
Real-time Fraud Alert Monitoring System
Continuously monitors transactions and generates alerts for suspicious activity
"""
import sys
import os
import time
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import threading

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.alerting_system import FraudAlertSystem
from src.data_preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeFraudMonitor:
    def __init__(self, model_path='models/logistic_regression.pkl', 
                 db_path='fraud_detection.db',
                 alert_config='alert_config.json',
                 check_interval=30):
        """
        Initialize real-time fraud monitor
        
        Args:
            model_path: Path to trained model
            db_path: Path to SQLite database
            alert_config: Path to alert configuration
            check_interval: Seconds between monitoring cycles
        """
        self.model_path = model_path
        self.db_path = db_path
        self.check_interval = check_interval
        self.running = False
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.alert_system = FraudAlertSystem(alert_config)
        
        # Load trained model
        self.model = self.load_model()
        
        # Setup database
        self.setup_database()
        
        logger.info("Real-time fraud monitor initialized")
    
    def load_model(self):
        """Load the trained fraud detection model"""
        try:
            model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def setup_database(self):
        """Setup database tables for alerts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create fraud_alerts table if it doesn't exist
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
            
            # Create monitoring_log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_timestamp TEXT,
                    transactions_checked INTEGER,
                    alerts_generated INTEGER,
                    status TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Database tables setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
    
    def get_new_transactions(self, last_check_time=None):
        """
        Get new transactions since last check
        
        Args:
            last_check_time: datetime of last check
            
        Returns:
            DataFrame of new transactions
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            if last_check_time:
                # Get transactions since last check (using Transaction_Date column)
                query = """
                    SELECT * FROM transactions 
                    WHERE Transaction_Date > ? 
                    ORDER BY Transaction_Date DESC
                """
                df = pd.read_sql(query, conn, params=[last_check_time.isoformat()])
            else:
                # Get recent transactions (last 100) (using Transaction_Date column)
                query = """
                    SELECT * FROM transactions 
                    ORDER BY Transaction_Date DESC 
                    LIMIT 100
                """
                df = pd.read_sql(query, conn)
            
            conn.close()
            
            # Normalize column names to match expected format
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
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting new transactions: {e}")
            return pd.DataFrame()
    
    def preprocess_transaction(self, transaction_df):
        """
        Preprocess transaction data for model prediction
        
        Args:
            transaction_df: DataFrame of transactions
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        try:
            # Handle missing values
            df_cleaned = self.preprocessor.handle_missing_values(transaction_df)
            
            # Encode categorical features
            categorical_columns = ['location', 'device', 'merchant_category', 'payment_method']
            categorical_columns = [col for col in categorical_columns if col in df_cleaned.columns]
            if categorical_columns:
                df_cleaned = self.preprocessor.encode_categorical_features(df_cleaned, categorical_columns)
            
            # Feature engineering
            df_cleaned = self.preprocessor.feature_engineering(df_cleaned)
            
            # Normalize features
            numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            exclude_columns = ['transaction_id', 'customer_id', 'is_fraud']
            numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
            if numeric_columns:
                df_cleaned = self.preprocessor.normalize_features(df_cleaned, numeric_columns)
            
            # Handle any remaining NaN values
            df_cleaned = df_cleaned.fillna(0)
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error preprocessing transactions: {e}")
            return None
    
    def predict_fraud_probability(self, preprocessed_df):
        """
        Predict fraud probability for transactions
        
        Args:
            preprocessed_df: Preprocessed DataFrame
            
        Returns:
            Array of fraud probabilities
        """
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return np.zeros(len(preprocessed_df))
            
            # Prepare features (remove non-feature columns)
            feature_columns = [col for col in preprocessed_df.columns 
                             if col not in ['transaction_id', 'customer_id', 'timestamp', 'is_fraud']]
            
            X = preprocessed_df[feature_columns]
            
            # Get fraud probabilities
            if hasattr(self.model, 'predict_proba'):
                fraud_probabilities = self.model.predict_proba(X)[:, 1]
            else:
                # For models without predict_proba, use decision function or predictions
                if hasattr(self.model, 'decision_function'):
                    decisions = self.model.decision_function(X)
                    # Convert to probabilities using sigmoid
                    fraud_probabilities = 1 / (1 + np.exp(-decisions))
                else:
                    predictions = self.model.predict(X)
                    fraud_probabilities = predictions.astype(float)
            
            return fraud_probabilities
            
        except Exception as e:
            logger.error(f"Error predicting fraud probability: {e}")
            return np.zeros(len(preprocessed_df))
    
    def generate_alert(self, transaction, fraud_probability):
        """
        Generate and store fraud alert
        
        Args:
            transaction: Transaction data
            fraud_probability: Predicted fraud probability
        """
        try:
            # Determine risk level
            if fraud_probability > 0.9:
                risk_level = "HIGH"
            elif fraud_probability > 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Store alert in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO fraud_alerts 
                (transaction_id, amount, location, fraud_probability, alert_timestamp, model_used, risk_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                transaction.get('transaction_id', 'N/A'),
                transaction.get('amount', 0),
                transaction.get('location', 'N/A'),
                fraud_probability,
                datetime.now().isoformat(),
                self.model_path.split('/')[-1],
                risk_level
            ))
            
            conn.commit()
            conn.close()
            
            # Prepare transaction details for alert notification
            transaction_details = {
                'transaction_id': transaction.get('transaction_id', 'N/A'),
                'amount': transaction.get('amount', 0),
                'location': transaction.get('location', 'N/A'),
                'fraud_probability': fraud_probability,
                'timestamp': transaction.get('timestamp', datetime.now().isoformat())
            }
            
            # Send alert notification
            alert_sent = self.alert_system.send_alert(transaction_details)
            
            logger.info(f"Alert generated for transaction {transaction.get('transaction_id')} - "
                       f"Risk: {risk_level}, Probability: {fraud_probability:.2%}, Notified: {alert_sent}")
            
            return alert_sent
            
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
            return False
    
    def monitor_cycle(self):
        """Execute one monitoring cycle"""
        try:
            check_time = datetime.now()
            logger.info(f"Starting monitoring cycle at {check_time}")
            
            # Get new transactions
            transactions_df = self.get_new_transactions()
            
            if transactions_df.empty:
                logger.info("No new transactions to monitor")
                self.log_monitoring_cycle(0, 0, "No transactions")
                return
            
            logger.info(f"Checking {len(transactions_df)} transactions")
            
            # Preprocess transactions
            preprocessed_df = self.preprocess_transaction(transactions_df)
            
            if preprocessed_df is None or preprocessed_df.empty:
                logger.error("Error preprocessing transactions")
                self.log_monitoring_cycle(len(transactions_df), 0, "Preprocessing error")
                return
            
            # Predict fraud probabilities
            fraud_probabilities = self.predict_fraud_probability(preprocessed_df)
            
            # Generate alerts for high-risk transactions
            alerts_generated = 0
            threshold = self.alert_system.config.get('threshold', 0.8)
            
            for i, (idx, transaction) in enumerate(transactions_df.iterrows()):
                fraud_prob = fraud_probabilities[i]
                
                if fraud_prob >= threshold:
                    # Convert transaction to dict for alert generation
                    transaction_dict = transaction.to_dict()
                    self.generate_alert(transaction_dict, fraud_prob)
                    alerts_generated += 1
            
            logger.info(f"Monitoring cycle completed: {len(transactions_df)} transactions checked, "
                       f"{alerts_generated} alerts generated")
            
            self.log_monitoring_cycle(len(transactions_df), alerts_generated, "Success")
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
            self.log_monitoring_cycle(0, 0, f"Error: {str(e)}")
    
    def log_monitoring_cycle(self, transactions_checked, alerts_generated, status):
        """Log monitoring cycle to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO monitoring_log 
                (check_timestamp, transactions_checked, alerts_generated, status)
                VALUES (?, ?, ?, ?)
            """, (datetime.now().isoformat(), transactions_checked, alerts_generated, status))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging monitoring cycle: {e}")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        self.running = True
        logger.info("Starting real-time fraud monitoring...")
        
        while self.running:
            try:
                self.monitor_cycle()
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        logger.info("Stopping real-time fraud monitoring...")

def main():
    """Main function to run real-time monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Fraud Alert Monitoring')
    parser.add_argument('--model_path', type=str, default='models/logistic_regression.pkl',
                       help='Path to trained model')
    parser.add_argument('--db_path', type=str, default='fraud_detection.db',
                       help='Path to SQLite database')
    parser.add_argument('--alert_config', type=str, default='alert_config.json',
                       help='Path to alert configuration')
    parser.add_argument('--check_interval', type=int, default=30,
                       help='Seconds between monitoring cycles')
    parser.add_argument('--single_cycle', action='store_true',
                       help='Run single monitoring cycle instead of continuous')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = RealTimeFraudMonitor(
        model_path=args.model_path,
        db_path=args.db_path,
        alert_config=args.alert_config,
        check_interval=args.check_interval
    )
    
    if args.single_cycle:
        # Run single monitoring cycle
        logger.info("Running single monitoring cycle...")
        monitor.monitor_cycle()
        logger.info("Single cycle completed")
    else:
        # Start continuous monitoring
        try:
            monitor.start_monitoring()
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            logger.info("Monitoring stopped")

if __name__ == "__main__":
    main()