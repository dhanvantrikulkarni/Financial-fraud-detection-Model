import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_fraud_data(num_samples=10000, fraud_ratio=0.1):
    np.random.seed(42)
    random.seed(42)
    
    # Generate customer IDs
    num_customers = int(num_samples * 0.3)  # 30% of transactions are from repeat customers
    customer_ids = [f"CUST_{i:05d}" for i in range(num_customers)]
    
    # Generate locations
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
                'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
                'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Washington']
    
    # Generate devices
    devices = ['iPhone', 'Android', 'Windows PC', 'Mac', 'Tablet', 'Other']
    
    # Generate timestamps over the past 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = []
    
    for i in range(num_samples):
        # Determine if this is a fraudulent transaction
        is_fraud = 1 if random.random() < fraud_ratio else 0
        
        # Generate transaction ID
        transaction_id = i + 1
        
        # Select customer ID (fraudulent transactions often from new customers)
        if is_fraud and random.random() < 0.7:
            customer_id = f"CUST_NEW_{random.randint(10000, 99999)}"
        else:
            customer_id = random.choice(customer_ids)
        
        # Generate amount (fraudulent transactions often have higher amounts)
        if is_fraud:
            amount = np.random.exponential(scale=500) + 100  # Higher amounts for fraud
        else:
            amount = np.random.exponential(scale=50) + 10  # Normal amounts
        
        amount = min(amount, 10000)  # Cap at $10,000
        
        # Generate timestamp
        timestamp = start_date + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Generate location (fraudulent transactions often from unusual locations)
        if is_fraud and random.random() < 0.5:
            location = random.choice(locations[-5:])  # Less common locations
        else:
            location = random.choice(locations)
        
        # Generate device
        device = random.choice(devices)
        
        data.append({
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'amount': round(amount, 2),
            'timestamp': timestamp,
            'location': location,
            'device': device,
            'is_fraud': is_fraud
        })
    
    df = pd.DataFrame(data)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def generate_realistic_fraud_patterns(num_samples=10000):
    """Generate data with more realistic fraud patterns"""
    np.random.seed(42)
    random.seed(42)
    
    # Generate legitimate transaction patterns
    legitimate_patterns = []
    
    # Pattern 1: Regular small transactions from same customer
    for i in range(3000):
        customer_id = f"CUST_REG_{i % 500}"
        legitimate_patterns.append({
            'transaction_id': len(legitimate_patterns) + 1,
            'customer_id': customer_id,
            'amount': np.random.normal(50, 20),
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 30)),
            'location': random.choice(['New York', 'Los Angeles', 'Chicago']),
            'device': 'iPhone',
            'is_fraud': 0
        })
    
    # Pattern 2: Medium transactions from various customers
    for i in range(4000):
        customer_id = f"CUST_VAR_{i % 800}"
        legitimate_patterns.append({
            'transaction_id': len(legitimate_patterns) + 1,
            'customer_id': customer_id,
            'amount': np.random.normal(200, 50),
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 30)),
            'location': random.choice(['Houston', 'Phoenix', 'Philadelphia']),
            'device': random.choice(['Android', 'Windows PC']),
            'is_fraud': 0
        })
    
    # Generate fraudulent patterns
    fraud_patterns = []
    
    # Pattern 1: High amount transactions from new customers
    for i in range(500):
        fraud_patterns.append({
            'transaction_id': len(legitimate_patterns) + len(fraud_patterns) + 1,
            'customer_id': f"CUST_NEW_{random.randint(10000, 99999)}",
            'amount': np.random.normal(2000, 500),
            'timestamp': datetime.now() - timedelta(hours=random.randint(0, 24)),
            'location': random.choice(['San Diego', 'Dallas', 'San Jose']),
            'device': 'Other',
            'is_fraud': 1
        })
    
    # Pattern 2: Rapid multiple transactions from same customer
    base_customer = f"CUST_SUSPICIOUS_{random.randint(1000, 9999)}"
    for i in range(200):
        fraud_patterns.append({
            'transaction_id': len(legitimate_patterns) + len(fraud_patterns) + 1,
            'customer_id': base_customer,
            'amount': np.random.normal(500, 100),
            'timestamp': datetime.now() - timedelta(minutes=random.randint(0, 60)),
            'location': 'Austin',
            'device': 'Android',
            'is_fraud': 1
        })
    
    # Pattern 3: Transactions from unusual locations
    for i in range(300):
        fraud_patterns.append({
            'transaction_id': len(legitimate_patterns) + len(fraud_patterns) + 1,
            'customer_id': f"CUST_{random.randint(1, 1000)}",
            'amount': np.random.normal(300, 80),
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 7)),
            'location': random.choice(['Charlotte', 'San Francisco', 'Indianapolis']),
            'device': 'Tablet',
            'is_fraud': 1
        })
    
    # Combine and create DataFrame
    all_data = legitimate_patterns + fraud_patterns
    df = pd.DataFrame(all_data)
    
    # Ensure amount is positive
    df['amount'] = df['amount'].abs()
    df['amount'] = df['amount'].round(2)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Generate sample data
    print("Generating basic sample data...")
    df_basic = generate_fraud_data(num_samples=10000, fraud_ratio=0.1)
    print(f"Generated {len(df_basic)} samples")
    print(f"Fraud count: {df_basic['is_fraud'].sum()}")
    print(f"Fraud rate: {df_basic['is_fraud'].mean():.2%}")
    
    # Save to CSV
    df_basic.to_csv('data/sample_fraud_data.csv', index=False)
    print("Saved to data/sample_fraud_data.csv")
    
    # Generate realistic patterns
    print("\nGenerating realistic fraud patterns...")
    df_realistic = generate_realistic_fraud_patterns(num_samples=10000)
    print(f"Generated {len(df_realistic)} samples")
    print(f"Fraud count: {df_realistic['is_fraud'].sum()}")
    print(f"Fraud rate: {df_realistic['is_fraud'].mean():.2%}")
    
    # Save to CSV
    df_realistic.to_csv('data/realistic_fraud_data.csv', index=False)
    print("Saved to data/realistic_fraud_data.csv")
