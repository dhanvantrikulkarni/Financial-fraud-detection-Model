"""
Quick start script for Financial Fraud Detection Model
This script helps users get started quickly with the system
"""
import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'streamlit', 
        'plotly', 'joblib', 'xgboost', 'lightgbm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            logger.warning(f"✗ {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Please run: pip install -r requirements.txt")
        return False
    
    logger.info("All required dependencies are installed!")
    return True

def setup_directories():
    """Create necessary directories"""
    logger.info("Setting up directories...")
    
    directories = ['data', 'models', 'streaming_input', 'streaming_output']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory already exists: {directory}")

def generate_sample_data():
    """Generate sample data for testing"""
    logger.info("Generating sample data...")
    
    try:
        from src.generate_sample_data import generate_fraud_data
        from src.data_preprocessing import DataPreprocessor
        
        # Generate data
        df = generate_fraud_data(num_samples=10000, fraud_ratio=0.1)
        logger.info(f"Generated {len(df)} samples")
        logger.info(f"Fraud count: {df['is_fraud'].sum()}")
        logger.info(f"Fraud rate: {df['is_fraud'].mean():.2%}")
        
        # Save to CSV
        df.to_csv('data/sample_fraud_data.csv', index=False)
        logger.info("Saved to data/sample_fraud_data.csv")
        
        # Process and save to database
        preprocessor = DataPreprocessor()
        preprocessor.etl_process(df)
        logger.info("Data saved to database")
        
        return True
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return False

def train_basic_models():
    """Train basic models for quick start"""
    logger.info("Training basic models...")
    
    try:
        # Run training script with basic options
        result = subprocess.run([
            sys.executable, 'train_models.py',
            '--data_path', 'data/sample_fraud_data.csv',
            '--model_type', 'supervised',
            '--supervised_models', 'logistic_regression', 'random_forest'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Basic models trained successfully!")
            logger.info(result.stdout)
            return True
        else:
            logger.error("Error training models:")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"Error training models: {e}")
        return False

def start_dashboard():
    """Start the Streamlit dashboard"""
    logger.info("Starting Streamlit dashboard...")
    logger.info("Dashboard will be available at http://localhost:8501")
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dashboard/app.py'])
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")

def main():
    """Main quick start function"""
    print("=" * 60)
    print("Financial Fraud Detection Model - Quick Start")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Please install dependencies first: pip install -r requirements.txt")
        return
    
    # Setup directories
    setup_directories()
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Generate sample data")
    print("2. Train basic models")
    print("3. Start dashboard")
    print("4. Full setup (data + models)")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        generate_sample_data()
    elif choice == '2':
        train_basic_models()
    elif choice == '3':
        start_dashboard()
    elif choice == '4':
        if generate_sample_data():
            if train_basic_models():
                print("\n✓ Full setup completed successfully!")
                print("You can now start the dashboard with: streamlit run dashboard/app.py")
    elif choice == '5':
        print("Goodbye!")
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
