import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sqlite3
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.column_mapping = {
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
        
    def normalize_column_names(self, df):
        """Normalize column names to match expected format"""
        df = df.copy()
        
        # Check if the original column names exist and map them
        original_columns = df.columns.tolist()
        column_mapping = {}
        
        for old_col in original_columns:
            if old_col in self.column_mapping:
                column_mapping[old_col] = self.column_mapping[old_col]
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            logger.info(f"Renamed columns: {column_mapping}")
        
        return df
    
    def load_data(self, file_path):
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
            
            # Normalize column names
            df = self.normalize_column_names(df)
            
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, df):
        df = df.copy()
        
        # Numerical columns - fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Categorical columns - fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_value = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
            df[col] = df[col].fillna(mode_value)
        
        logger.info("Missing values handled")
        return df
    
    def encode_categorical_features(self, df, categorical_columns):
        df = df.copy()
        
        # Add categorical columns from real dataset if they exist
        additional_categorical = ['merchant_category', 'payment_method']
        for col in additional_categorical:
            if col in df.columns and col not in categorical_columns:
                categorical_columns.append(col)
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen labels by re-fitting the encoder with new data
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except ValueError as e:
                    if "previously unseen labels" in str(e):
                        logger.warning(f"Unseen labels detected in {col}, re-fitting encoder")
                        self.label_encoders[col] = LabelEncoder()
                        df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                    else:
                        raise
        
        logger.info(f"Categorical features encoded: {categorical_columns}")
        return df
    
    def feature_engineering(self, df):
        df = df.copy()
        
        # Extract temporal features if timestamp exists
        timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'Transaction_Date' if 'Transaction_Date' in df.columns else None
        if timestamp_col:
            # Try different date formats
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%d-%m-%Y %H:%M')
            except:
                try:
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%m-%d-%Y %H:%M')
                except:
                    try:
                        df[timestamp_col] = pd.to_datetime(df[timestamp_col], dayfirst=True)
                    except:
                        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='mixed')
            
            # Rename to standardized 'timestamp' if needed
            if timestamp_col != 'timestamp':
                df['timestamp'] = df[timestamp_col]
            
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Transaction frequency features
        if 'customer_id' in df.columns:
            customer_counts = df['customer_id'].value_counts()
            df['customer_transaction_count'] = df['customer_id'].map(customer_counts)
        
        # Amount-based features
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            df['amount_squared'] = df['amount'] ** 2
        
        # Additional features from real dataset
        if 'merchant_category' in df.columns:
            # Keep merchant_category as categorical (will be encoded later)
            pass
        
        if 'payment_method' in df.columns:
            # Keep payment_method as categorical (will be encoded later)
            pass
        
        if 'is_international' in df.columns:
            # Ensure is_international is numeric
            df['is_international'] = pd.to_numeric(df['is_international'], errors='coerce').fillna(0)
        
        if 'previous_transactions' in df.columns:
            # Ensure previous_transactions is numeric
            df['previous_transactions'] = pd.to_numeric(df['previous_transactions'], errors='coerce').fillna(0)
        
        if 'average_spend' in df.columns:
            # Ensure average_spend is numeric
            df['average_spend'] = pd.to_numeric(df['average_spend'], errors='coerce').fillna(0)
        
        if 'account_age_days' in df.columns:
            # Ensure account_age_days is numeric
            df['account_age_days'] = pd.to_numeric(df['account_age_days'], errors='coerce').fillna(0)
        
        if 'suspicious_keyword' in df.columns:
            # Convert suspicious_keyword to binary (Yes=1, No=0)
            df['suspicious_keyword'] = df['suspicious_keyword'].map({'Yes': 1, 'No': 0}).fillna(0)
        
        logger.info("Feature engineering completed")
        return df
    
    def normalize_features(self, df, numeric_columns):
        df = df.copy()
        
        # Filter to only include columns that exist and are numeric
        valid_numeric_columns = []
        for col in numeric_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                valid_numeric_columns.append(col)
        
        if self.feature_columns is None:
            self.feature_columns = valid_numeric_columns
        
        if valid_numeric_columns:
            df[valid_numeric_columns] = self.scaler.fit_transform(df[valid_numeric_columns])
            logger.info(f"Features normalized: {valid_numeric_columns}")
        else:
            logger.warning("No valid numeric columns found for normalization")
        
        return df
    
    def prepare_training_data(self, df, target_column='is_fraud', test_size=0.2, random_state=42):
        # Check if target column exists, try alternative names
        if target_column not in df.columns:
            # Try alternative target column names
            alternative_targets = ['Fraudulent', 'fraud', 'is_fraudulent']
            for alt_target in alternative_targets:
                if alt_target in df.columns:
                    target_column = alt_target
                    logger.info(f"Using alternative target column: {target_column}")
                    break
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe. Available columns: {df.columns.tolist()}")
        
        # Remove non-feature columns (include both normalized and original names)
        columns_to_drop = [target_column]
        non_feature_columns = [
            'transaction_id', 'customer_id', 'timestamp',  # normalized names
            'Transaction_ID', 'Customer_ID', 'Transaction_Date'  # original names
        ]
        for col in non_feature_columns:
            if col in df.columns:
                columns_to_drop.append(col)
        
        # Also drop any remaining string/object columns that might be IDs
        for col in df.columns:
            if col not in columns_to_drop and df[col].dtype == 'object':
                # Check if this looks like an ID column (high cardinality, mostly unique)
                if df[col].nunique() > len(df) * 0.9:  # More than 90% unique values
                    columns_to_drop.append(col)
                    logger.info(f"Dropping potential ID column: {col}")
        
        # Separate features and target
        X = df.drop(columns=columns_to_drop)
        y = df[target_column]
        
        # Ensure all features are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                logger.warning(f"Column {col} is still object type, attempting conversion")
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    logger.error(f"Could not convert column {col} to numeric, dropping it")
                    X = X.drop(columns=[col])
        
        # Fill any NaN values that resulted from conversion
        X = X.fillna(0)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        logger.info(f"Feature columns: {X.columns.tolist()}")
        return X_train, X_test, y_train, y_test
    
    def etl_process(self, df, db_path="fraud_detection.db"):
        try:
            # Clean the data
            df_cleaned = self.handle_missing_values(df)
            
            # Connect to SQLite database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create the transaction table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    amount REAL,
                    timestamp TEXT,
                    location TEXT,
                    device TEXT,
                    is_fraud INTEGER,
                    hour INTEGER,
                    day_of_week INTEGER,
                    is_weekend INTEGER,
                    amount_log REAL,
                    amount_squared REAL,
                    merchant_category TEXT,
                    payment_method TEXT,
                    is_international INTEGER,
                    previous_transactions INTEGER,
                    average_spend REAL,
                    account_age_days INTEGER,
                    suspicious_keyword INTEGER
                )
            """)
            
            # Insert the cleaned data
            df_cleaned.to_sql("transactions", conn, if_exists="replace", index=False)
            
            conn.commit()
            conn.close()
            
            logger.info("ETL process completed successfully")
            return True
        except Exception as e:
            logger.error(f"ETL process failed: {e}")
            return False
