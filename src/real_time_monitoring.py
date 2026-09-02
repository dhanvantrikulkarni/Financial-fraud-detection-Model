import json
import logging
import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
import joblib
import sqlite3
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeFraudMonitor:
    def __init__(self, model_path=None, model_type='supervised'):
        self.model = None
        self.model_type = model_type
        self.consumer = None
        self.producer = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def setup_kafka_consumer(self, topic='transaction_stream', bootstrap_servers=['localhost:9092']):
        try:
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest'
            )
            logger.info(f"Kafka consumer connected to topic: {topic}")
        except Exception as e:
            logger.error(f"Error setting up Kafka consumer: {e}")
            raise
    
    def setup_kafka_producer(self, bootstrap_servers=['localhost:9092']):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            logger.info("Kafka producer initialized")
        except Exception as e:
            logger.error(f"Error setting up Kafka producer: {e}")
            raise
    
    def process_transaction(self, transaction):
        try:
            # Extract features from transaction
            features = self.extract_features(transaction)
            
            # Make prediction
            if self.model:
                prediction = self.model.predict([features])[0]
                probability = self.model.predict_proba([features])[0][1] if hasattr(self.model, 'predict_proba') else 0.0
            else:
                prediction = 0
                probability = 0.0
            
            result = {
                'transaction_id': transaction.get('transaction_id'),
                'prediction': int(prediction),
                'fraud_probability': float(probability),
                'timestamp': datetime.now().isoformat(),
                'amount': transaction.get('amount'),
                'location': transaction.get('location')
            }
            
            return result
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
            return None
    
    def extract_features(self, transaction):
        # This is a simplified feature extraction
        # In production, this should match the training feature engineering
        features = [
            transaction.get('amount', 0),
            transaction.get('hour', 0),
            transaction.get('day_of_week', 0),
            transaction.get('is_weekend', 0),
            transaction.get('customer_transaction_count', 1),
            transaction.get('location_encoded', 0),
            transaction.get('device_encoded', 0)
        ]
        return features
    
    def monitor_stream(self, alert_callback=None):
        if not self.consumer:
            logger.error("Kafka consumer not initialized")
            return
        
        logger.info("Starting real-time fraud monitoring...")
        
        for message in self.consumer:
            try:
                transaction = message.value
                result = self.process_transaction(transaction)
                
                if result:
                    logger.info(f"Processed transaction {result['transaction_id']}: "
                              f"Prediction={result['prediction']}, "
                              f"Probability={result['fraud_probability']:.4f}")
                    
                    # Send alert if fraud detected
                    if result['prediction'] == 1 and alert_callback:
                        alert_callback(result)
                    
                    # Send result to alerts topic if producer exists
                    if self.producer:
                        self.producer.send('fraud_alerts', value=result)
                        self.producer.flush()
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    def save_to_database(self, result, db_path="fraud_detection.db"):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO fraud_alerts 
                (transaction_id, prediction, fraud_probability, timestamp, amount, location)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                result['transaction_id'],
                result['prediction'],
                result['fraud_probability'],
                result['timestamp'],
                result['amount'],
                result['location']
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Alert saved to database for transaction {result['transaction_id']}")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")

class SparkStreamingMonitor:
    def __init__(self, model_path=None):
        self.spark = None
        self.model = None
        self.model_path = model_path
        
    def setup_spark_session(self, app_name="FraudDetectionSystem"):
        try:
            from pyspark.sql import SparkSession
            from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, TimestampType
            
            self.spark = SparkSession.builder \
                .appName(app_name) \
                .getOrCreate()
            
            # Define schema
            self.schema = StructType([
                StructField("transaction_id", IntegerType(), True),
                StructField("customer_id", StringType(), True),
                StructField("amount", DoubleType(), True),
                StructField("timestamp", TimestampType(), True),
                StructField("location", StringType(), True),
                StructField("device", StringType(), True),
                StructField("is_fraud", IntegerType(), True)
            ])
            
            logger.info("Spark session created successfully")
        except Exception as e:
            logger.error(f"Error setting up Spark session: {e}")
            raise
    
    def load_spark_model(self):
        try:
            from pyspark.ml.classification import LogisticRegressionModel
            
            if self.model_path:
                self.model = LogisticRegressionModel.load(self.model_path)
                logger.info(f"Spark ML model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading Spark model: {e}")
    
    def process_stream(self, input_path="streaming_input/", output_path="streaming_output/"):
        if not self.spark:
            logger.error("Spark session not initialized")
            return
        
        try:
            from pyspark.sql.functions import hour, col
            from pyspark.ml.feature import VectorAssembler, StringIndexer
            
            # Read stream
            transactions = self.spark.readStream \
                .schema(self.schema) \
                .option("maxFilesPerTrigger", 1) \
                .csv(input_path)
            
            # Feature engineering
            transactions = transactions.withColumn("hour", hour(col("timestamp")))
            
            # Index categorical features
            location_indexer = StringIndexer(inputCol="location", outputCol="location_idx")
            device_indexer = StringIndexer(inputCol="device", outputCol="device_idx")
            
            indexed_df = location_indexer.fit(transactions).transform(transactions)
            indexed_df = device_indexer.fit(indexed_df).transform(indexed_df)
            
            # Assemble features
            assembler = VectorAssembler(
                inputCols=["amount", "hour", "location_idx", "device_idx"],
                outputCol="features"
            )
            feature_df = assembler.transform(indexed_df)
            
            # Make predictions if model is loaded
            if self.model:
                predictions = self.model.transform(feature_df)
                
                # Write stream with predictions
                query = predictions.writeStream \
                    .outputMode("append") \
                    .format("csv") \
                    .option("path", output_path) \
                    .option("checkpointLocation", "checkpoint/") \
                    .start()
            else:
                # Write stream without predictions
                query = feature_df.writeStream \
                    .outputMode("append") \
                    .format("csv") \
                    .option("path", output_path) \
                    .option("checkpointLocation", "checkpoint/") \
                    .start()
            
            logger.info("Stream processing started")
            return query
            
        except Exception as e:
            logger.error(f"Error processing stream: {e}")
            raise
    
    def stop_stream(self, query):
        if query:
            query.stop()
            logger.info("Stream stopped")
        
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")
