"""
Test script to verify email and Slack notification configuration
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.alerting_system import FraudAlertSystem

def test_notifications():
    """Test email and Slack notification setup"""
    
    print("Testing Fraud Alert Notification System...")
    print("=" * 50)
    
    # Initialize alert system
    alert_system = FraudAlertSystem('alert_config.json')
    
    # Create test transaction details
    test_transaction = {
        'transaction_id': 'TEST_001',
        'amount': 9999.99,
        'location': 'Unknown Location',
        'fraud_probability': 0.95,
        'timestamp': '2026-09-02 14:30:00'
    }
    
    print(f"\nTest Transaction Details:")
    print(f"  Transaction ID: {test_transaction['transaction_id']}")
    print(f"  Amount: ${test_transaction['amount']:.2f}")
    print(f"  Location: {test_transaction['location']}")
    print(f"  Fraud Probability: {test_transaction['fraud_probability']:.2%}")
    print(f"  Timestamp: {test_transaction['timestamp']}")
    
    # Check configuration
    print(f"\nCurrent Configuration:")
    print(f"  Email Enabled: {alert_system.config['email']['enabled']}")
    print(f"  Slack Enabled: {alert_system.config['slack']['enabled']}")
    print(f"  Alert Threshold: {alert_system.config['threshold']:.2%}")
    
    # Test email notification
    if alert_system.config['email']['enabled']:
        print(f"\nTesting Email Notification...")
        email_result = alert_system.send_email_alert(test_transaction)
        if email_result:
            print(f"  Email notification sent successfully!")
        else:
            print(f"  Email notification failed. Check your credentials.")
    else:
        print(f"\nEmail notifications are disabled. Set 'enabled': true in alert_config.json")
    
    # Test Slack notification
    if alert_system.config['slack']['enabled']:
        print(f"\nTesting Slack Notification...")
        slack_result = alert_system.send_slack_alert(test_transaction)
        if slack_result:
            print(f"  Slack notification sent successfully!")
        else:
            print(f"  Slack notification failed. Check your webhook URL.")
    else:
        print(f"\nSlack notifications are disabled. Set 'enabled': true in alert_config.json")
    
    # Test combined alert
    print(f"\nTesting Combined Alert System...")
    combined_result = alert_system.send_alert(test_transaction)
    if combined_result:
        print(f"  Alert sent successfully through enabled channels!")
    else:
        print(f"  No notifications sent. Enable at least one channel.")
    
    print(f"\n" + "=" * 50)
    print(f"Test completed!")
    print(f"\nNext steps:")
    print(f"1. If notifications failed, check your credentials in alert_config.json")
    print(f"2. Run real-time monitoring: python run_real_time_alerts.py")
    print(f"3. Or generate initial alerts: python generate_initial_alerts.py")

if __name__ == "__main__":
    test_notifications()