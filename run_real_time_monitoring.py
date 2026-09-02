"""
Real-time fraud monitoring script using Kafka
"""
import sys
import os
import argparse
import logging
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.real_time_monitoring import RealTimeFraudMonitor
from src.alerting_system import FraudAlertSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def alert_callback(transaction_result):
    """Callback function for handling fraud alerts"""
    logger.info(f"FRAUD ALERT: Transaction {transaction_result['transaction_id']} "
               f"detected as fraudulent with probability {transaction_result['fraud_probability']:.2%}")
    
    # Initialize alert system
    alert_system = FraudAlertSystem()
    
    # Send alert
    alert_system.send_alert(transaction_result)

def main():
    parser = argparse.ArgumentParser(description='Run Real-time Fraud Monitoring')
    parser.add_argument('--model_path', type=str, default='models/logistic_regression.pkl',
                       help='Path to trained model file')
    parser.add_argument('--model_type', type=str, default='supervised',
                       choices=['supervised', 'unsupervised'],
                       help='Type of model to use')
    parser.add_argument('--kafka_topic', type=str, default='transaction_stream',
                       help='Kafka topic to consume from')
    parser.add_argument('--kafka_servers', nargs='+', default=['localhost:9092'],
                       help='Kafka bootstrap servers')
    parser.add_argument('--enable_alerts', action='store_true',
                       help='Enable email and Slack alerts')
    
    args = parser.parse_args()
    
    logger.info("Starting Real-time Fraud Monitoring")
    
    # Initialize monitor
    monitor = RealTimeFraudMonitor(
        model_path=args.model_path,
        model_type=args.model_type
    )
    
    # Setup Kafka consumer
    monitor.setup_kafka_consumer(
        topic=args.kafka_topic,
        bootstrap_servers=args.kafka_servers
    )
    
    # Setup Kafka producer for alerts
    monitor.setup_kafka_producer(
        bootstrap_servers=args.kafka_servers
    )
    
    # Start monitoring
    callback = alert_callback if args.enable_alerts else None
    monitor.monitor_stream(alert_callback=callback)

if __name__ == "__main__":
    main()
