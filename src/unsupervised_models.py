import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Autoencoder models will be disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnsupervisedFraudDetector:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def train_isolation_forest(self, X_train, contamination=0.1, **params):
        logger.info("Training Isolation Forest model")
        
        default_params = {
            'contamination': contamination,
            'random_state': 42,
            'n_estimators': 100
        }
        default_params.update(params)
        
        model = IsolationForest(**default_params)
        model.fit(X_train)
        
        self.models['isolation_forest'] = model
        logger.info("Isolation Forest training completed")
        return model
    
    def train_one_class_svm(self, X_train, nu=0.1, **params):
        logger.info("Training One-Class SVM model")
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['one_class_svm'] = scaler
        
        default_params = {
            'nu': nu,
            'kernel': 'rbf',
            'gamma': 'scale'
        }
        default_params.update(params)
        
        model = OneClassSVM(**default_params)
        model.fit(X_train_scaled)
        
        self.models['one_class_svm'] = model
        logger.info("One-Class SVM training completed")
        return model
    
    def train_autoencoder(self, X_train, encoding_dim=14, epochs=50, batch_size=32):
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is not available. Cannot train Autoencoder model.")
            return None
        
        logger.info("Training Autoencoder model")
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['autoencoder'] = scaler
        
        input_dim = X_train.shape[1]
        
        # Build autoencoder
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation="relu")(input_layer)
        encoder = Dense(encoding_dim // 2, activation="relu")(encoder)
        decoder = Dense(encoding_dim, activation="relu")(encoder)
        decoder = Dense(input_dim, activation="sigmoid")(decoder)
        
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train only on normal transactions (assuming fraud labels exist)
        # For pure unsupervised, train on all data
        autoencoder.fit(
            X_train_scaled, X_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=0
        )
        
        self.models['autoencoder'] = autoencoder
        logger.info("Autoencoder training completed")
        return autoencoder
    
    def predict_isolation_forest(self, X_test):
        model = self.models['isolation_forest']
        predictions = model.predict(X_test)
        # Convert -1 (anomaly) to 1 (fraud) and 1 (normal) to 0 (not fraud)
        predictions = (predictions == -1).astype(int)
        return predictions
    
    def predict_one_class_svm(self, X_test):
        model = self.models['one_class_svm']
        scaler = self.scalers['one_class_svm']
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        # Convert -1 (anomaly) to 1 (fraud) and 1 (normal) to 0 (not fraud)
        predictions = (predictions == -1).astype(int)
        return predictions
    
    def predict_autoencoder(self, X_test, threshold_percentile=95):
        model = self.models['autoencoder']
        scaler = self.scalers['autoencoder']
        X_test_scaled = scaler.transform(X_test)
        
        # Get reconstruction error
        reconstructions = model.predict(X_test_scaled, verbose=0)
        mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=1)
        
        # Determine threshold based on percentile
        threshold = np.percentile(mse, threshold_percentile)
        
        # Predict anomalies
        predictions = (mse > threshold).astype(int)
        return predictions, mse, threshold
    
    def evaluate_model(self, y_true, y_pred, model_name):
        logger.info(f"Evaluating {model_name}")
        
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        try:
            auc_score = roc_auc_score(y_true, y_pred)
        except:
            auc_score = 0.0
        
        results = {
            'classification_report': report,
            'confusion_matrix': cm,
            'auc_score': auc_score,
            'predictions': y_pred
        }
        
        logger.info(f"{model_name} - AUC Score: {auc_score:.4f}")
        return results
    
    def train_all_models(self, X_train):
        logger.info("Training all unsupervised models")
        
        self.train_isolation_forest(X_train)
        self.train_one_class_svm(X_train)
        self.train_autoencoder(X_train)
        
        logger.info("All unsupervised models trained successfully")
        return self.models
    
    def evaluate_all_models(self, X_test, y_test):
        results = {}
        
        # Isolation Forest
        if 'isolation_forest' in self.models:
            y_pred = self.predict_isolation_forest(X_test)
            results['isolation_forest'] = self.evaluate_model(y_test, y_pred, 'isolation_forest')
        
        # One-Class SVM
        if 'one_class_svm' in self.models:
            y_pred = self.predict_one_class_svm(X_test)
            results['one_class_svm'] = self.evaluate_model(y_test, y_pred, 'one_class_svm')
        
        # Autoencoder
        if 'autoencoder' in self.models:
            y_pred, mse, threshold = self.predict_autoencoder(X_test)
            results['autoencoder'] = self.evaluate_model(y_test, y_pred, 'autoencoder')
            results['autoencoder']['reconstruction_errors'] = mse
            results['autoencoder']['threshold'] = threshold
        
        return results
    
    def save_models(self, model_dir='models'):
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name == 'autoencoder':
                model_path = f"{model_dir}/{model_name}.keras"
                model.save(model_path)
                logger.info(f"Model saved: {model_path}")
            else:
                model_path = f"{model_dir}/{model_name}.pkl"
                joblib.dump(model, model_path)
                logger.info(f"Model saved: {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = f"{model_dir}/{scaler_name}_scaler.pkl"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved: {scaler_path}")
    
    def load_models(self, model_dir='models'):
        import os
        
        # Load Isolation Forest
        if os.path.exists(f"{model_dir}/isolation_forest.pkl"):
            self.models['isolation_forest'] = joblib.load(f"{model_dir}/isolation_forest.pkl")
            logger.info("Model loaded: isolation_forest")
        
        # Load One-Class SVM
        if os.path.exists(f"{model_dir}/one_class_svm.pkl"):
            self.models['one_class_svm'] = joblib.load(f"{model_dir}/one_class_svm.pkl")
            self.scalers['one_class_svm'] = joblib.load(f"{model_dir}/one_class_svm_scaler.pkl")
            logger.info("Model loaded: one_class_svm")
        
        # Load Autoencoder
        if os.path.exists(f"{model_dir}/autoencoder.keras"):
            self.models['autoencoder'] = tf.keras.models.load_model(f"{model_dir}/autoencoder.keras")
            self.scalers['autoencoder'] = joblib.load(f"{model_dir}/autoencoder_scaler.pkl")
            logger.info("Model loaded: autoencoder")
        
        return self.models
