import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. XGBoost models will be disabled.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. LightGBM models will be disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupervisedFraudDetector:
    def __init__(self):
        self.models = {}
        self.model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
        
    def train_logistic_regression(self, X_train, y_train, **params):
        logger.info("Training Logistic Regression model")
        
        # Handle NaN values
        if X_train.isnull().any().any():
            logger.warning("NaN values found in X_train, filling with 0")
            X_train = X_train.fillna(0)
        
        default_params = {
            'random_state': 42,
            'max_iter': 1000,
            'class_weight': 'balanced'
        }
        default_params.update(params)
        
        model = LogisticRegression(**default_params)
        model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = model
        logger.info("Logistic Regression training completed")
        return model
    
    def train_random_forest(self, X_train, y_train, **params):
        logger.info("Training Random Forest model")
        
        # Handle NaN values
        if X_train.isnull().any().any():
            logger.warning("NaN values found in X_train, filling with 0")
            X_train = X_train.fillna(0)
        
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        default_params.update(params)
        
        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        logger.info("Random Forest training completed")
        return model
    
    def train_xgboost(self, X_train, y_train, **params):
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost is not available. Cannot train XGBoost model.")
            return None
        
        logger.info("Training XGBoost model")
        
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'scale_pos_weight': (len(y_train) - sum(y_train)) / sum(y_train)
        }
        default_params.update(params)
        
        # Handle NaN values
        if X_train.isnull().any().any():
            logger.warning("NaN values found in X_train, filling with 0")
            X_train = X_train.fillna(0)
        
        model = xgb.XGBClassifier(**default_params)
        model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        logger.info("XGBoost training completed")
        return model
    
    def train_lightgbm(self, X_train, y_train, **params):
        if not LIGHTGBM_AVAILABLE:
            logger.error("LightGBM is not available. Cannot train LightGBM model.")
            return None
        
        logger.info("Training LightGBM model")
        
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'class_weight': 'balanced',
            'verbose': -1
        }
        default_params.update(params)
        
        # Handle NaN values
        if X_train.isnull().any().any():
            logger.warning("NaN values found in X_train, filling with 0")
            X_train = X_train.fillna(0)
        
        model = lgb.LGBMClassifier(**default_params)
        model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        logger.info("LightGBM training completed")
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        logger.info(f"Evaluating {model_name}")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results = {
            'classification_report': report,
            'confusion_matrix': cm,
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"{model_name} - AUC Score: {auc_score:.4f}")
        return results
    
    def train_all_models(self, X_train, y_train):
        logger.info("Training all supervised models")
        
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        
        logger.info("All supervised models trained successfully")
        return self.models
    
    def evaluate_all_models(self, X_test, y_test):
        results = {}
        
        for model_name in self.model_names:
            if model_name in self.models:
                results[model_name] = self.evaluate_model(
                    self.models[model_name], X_test, y_test, model_name
                )
        
        return results
    
    def save_models(self, model_dir='models'):
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = f"{model_dir}/{model_name}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Model saved: {model_path}")
    
    def load_models(self, model_dir='models'):
        import os
        
        for model_name in self.model_names:
            model_path = f"{model_dir}/{model_name}.pkl"
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"Model loaded: {model_path}")
        
        return self.models
    
    def get_best_model(self, X_test, y_test):
        best_model = None
        best_score = 0
        best_name = None
        
        for model_name in self.model_names:
            if model_name in self.models:
                y_pred_proba = self.models[model_name].predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                if auc_score > best_score:
                    best_score = auc_score
                    best_model = self.models[model_name]
                    best_name = model_name
        
        logger.info(f"Best model: {best_name} with AUC: {best_score:.4f}")
        return best_model, best_name, best_score
