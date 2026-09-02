# Financial Fraud Detection Model with Dashboard

A comprehensive machine learning system for detecting financial fraud in real-time with an interactive dashboard. This project implements both supervised and unsupervised learning approaches, real-time streaming capabilities, and automated alerting.

## 🚀 Features

### 1. Data Acquisition & Preprocessing
- **ETL Operations**: Extract, Transform, Load transactional data into SQLite database
- **Missing Value Handling**: Intelligent imputation for numerical and categorical data
- **Feature Engineering**: Temporal features, behavioral patterns, and anomaly extraction
- **Data Normalization**: StandardScaler for consistent feature scaling

### 2. Fraud Detection Algorithms

#### Supervised Learning Models
- **Logistic Regression**: Baseline classification with balanced class weights
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting with advanced regularization
- **LightGBM**: Fast gradient boosting framework

#### Unsupervised Learning Models
- **Isolation Forest**: Anomaly detection based on isolation
- **One-Class SVM**: Support vector method for outlier detection
- **Autoencoder**: Deep learning reconstruction error-based detection

### 3. Real-Time Fraud Monitoring
- **Apache Kafka Integration**: Stream processing of transaction data
- **PySpark Streaming**: Scalable real-time data processing
- **Instant Prediction**: Real-time fraud scoring on incoming transactions
- **Adaptive Learning**: Continuous model improvement mechanism

### 4. Interactive Dashboard
- **Real-time Visualization**: Live fraud detection metrics and trends
- **Model Performance**: Comprehensive evaluation metrics and comparisons
- **Transaction Analysis**: Detailed filtering and exploration capabilities
- **Alert Management**: Real-time fraud alert monitoring and history

### 5. Alerting System
- **Email Alerts**: Automated notifications for high-risk transactions
- **Slack Integration**: Real-time team notifications via webhooks
- **Configurable Thresholds**: Customizable risk score thresholds
- **Alert History**: Comprehensive logging of all fraud alerts

## 📋 Project Structure

```
Financial-fraud-detection-Model/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data loading, cleaning, and feature engineering
│   ├── supervised_models.py       # Supervised learning models (LR, RF, XGBoost, LightGBM)
│   ├── unsupervised_models.py     # Unsupervised models (Isolation Forest, SVM, Autoencoder)
│   ├── real_time_monitoring.py    # Kafka and Spark streaming integration
│   ├── alerting_system.py         # Email and Slack alerting
│   └── generate_sample_data.py    # Sample data generation for testing
├── dashboard/
│   └── app.py                      # Streamlit dashboard application
├── models/                         # Trained model storage
├── data/                           # Data storage directory
├── streaming_input/                # Input directory for streaming
├── streaming_output/              # Output directory for streaming
├── train_models.py                # Main training script
├── run_real_time_monitoring.py    # Real-time monitoring script
├── requirements.txt               # Python dependencies
├── alert_config.json             # Alert configuration
└── README.md                      # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Apache Kafka (for real-time monitoring)
- Java/Scala (for PySpark streaming)

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/dhanvantrikulkarni/Financial-fraud-detection-Model.git
cd Financial-fraud-detection-Model
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install additional dependencies for PySpark (optional)**
```bash
pip install pyspark
```

## 🚀 Usage

### 1. Train Models

**Generate sample data and train all models:**
```bash
python train_models.py --generate_data --num_samples 10000 --fraud_ratio 0.1 --model_type all
```

**Train with existing data:**
```bash
python train_models.py --data_path data/your_data.csv --model_type supervised
```

**Training options:**
- `--generate_data`: Generate sample data instead of loading from file
- `--num_samples`: Number of samples to generate (default: 10000)
- `--fraud_ratio`: Ratio of fraudulent transactions (default: 0.1)
- `--model_type`: Type of models to train (supervised/unsupervised/all)
- `--supervised_models`: Specific supervised models to train
- `--unsupervised_models`: Specific unsupervised models to train

### 2. Run Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`

**Dashboard Features:**
- **Dashboard Overview**: Key metrics, fraud statistics, and visualizations
- **Model Performance**: Compare model performance with detailed metrics
- **Transaction Analysis**: Filter and explore transaction data
- **Fraud Alerts**: View recent fraud alerts and alert history
- **Model Training**: Train and evaluate models directly from the dashboard
- **Data Upload**: Upload your own data or generate sample data

### 3. Real-Time Monitoring

**Start Kafka (if not already running):**
```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka
bin/kafka-server-start.sh config/server.properties

# Create topic
bin/kafka-topics.sh --create --topic transaction_stream --bootstrap-server localhost:9092
```

**Run real-time monitoring:**
```bash
python run_real_time_monitoring.py --model_path models/logistic_regression.pkl --enable_alerts
```

**Monitoring options:**
- `--model_path`: Path to trained model (default: models/logistic_regression.pkl)
- `--model_type`: Type of model (supervised/unsupervised)
- `--kafka_topic`: Kafka topic to consume from (default: transaction_stream)
- `--kafka_servers`: Kafka bootstrap servers (default: localhost:9092)
- `--enable_alerts`: Enable email and Slack alerts

### 4. Configure Alerts

Edit `alert_config.json` to configure your alert settings:

```json
{
  "email": {
    "enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your_email@gmail.com",
    "sender_password": "your_app_password",
    "recipients": ["security@yourcompany.com"]
  },
  "slack": {
    "enabled": true,
    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
  },
  "threshold": 0.8
}
```

**Note**: For Gmail, use an App Password instead of your regular password.

## 📊 Model Performance

The system includes multiple algorithms with different strengths:

### Supervised Models
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Robust ensemble with feature importance
- **XGBoost**: High performance with regularization
- **LightGBM**: Fast training with good accuracy

### Unsupervised Models
- **Isolation Forest**: Effective for anomaly detection
- **One-Class SVM**: Good for outlier detection
- **Autoencoder**: Deep learning approach for complex patterns

## 🔧 Advanced Configuration

### PySpark Streaming

For large-scale real-time processing:

```python
from src.real_time_monitoring import SparkStreamingMonitor

monitor = SparkStreamingMonitor()
monitor.setup_spark_session()
monitor.load_spark_model()
query = monitor.process_stream("streaming_input/", "streaming_output/")
```

### Custom Feature Engineering

Extend the `DataPreprocessor` class to add custom features:

```python
class CustomPreprocessor(DataPreprocessor):
    def custom_feature_engineering(self, df):
        # Add your custom features
        df['custom_feature'] = df['amount'] / df['customer_transaction_count']
        return df
```

## 📈 Evaluation Metrics

The system evaluates models using:
- **AUC-ROC Score**: Area under the receiver operating characteristic curve
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to detect all fraudulent transactions
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results

## 🔒 Security Considerations

- Never commit sensitive credentials (API keys, passwords) to the repository
- Use environment variables for sensitive configuration
- Implement proper authentication for the dashboard in production
- Regularly update dependencies for security patches
- Use HTTPS for all web communications in production

## 🚧 Future Enhancements

- [ ] AI-driven risk scoring for customer creditworthiness
- [ ] Blockchain-based fraud prevention system
- [ ] Mobile app integration for instant fraud alerts
- [ ] Graph neural networks for relationship detection
- [ ] Multi-modal data integration (text, images, transaction data)
- [ ] Federated learning for privacy-preserving fraud detection

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Support

For support, please open an issue in the GitHub repository or contact the development team.

## 🙏 Acknowledgments

- Scikit-learn for machine learning algorithms
- XGBoost and LightGBM for gradient boosting frameworks
- Streamlit for the interactive dashboard
- Apache Kafka for real-time streaming
- TensorFlow for deep learning models

---

**Note**: This is a demonstration project. For production use, additional security measures, scalability considerations, and compliance requirements should be implemented.
