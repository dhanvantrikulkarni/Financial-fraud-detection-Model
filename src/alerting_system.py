import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import json
from datetime import datetime
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudAlertSystem:
    def __init__(self, config_file='alert_config.json'):
        self.config = self.load_config(config_file)
        self.alert_history = []
        
    def load_config(self, config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Alert configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found. Using default configuration.")
            return self.get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        return {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': '',
                'sender_password': '',
                'recipients': []
            },
            'slack': {
                'enabled': False,
                'webhook_url': ''
            },
            'threshold': 0.8
        }
    
    def send_email_alert(self, transaction_details):
        if not self.config['email']['enabled']:
            logger.info("Email alerts are disabled")
            return False
        
        try:
            email_config = self.config['email']
            
            # Create message
            message = MIMEMultipart()
            message['From'] = email_config['sender_email']
            message['To'] = ', '.join(email_config['recipients'])
            message['Subject'] = f"🚨 FRAUD ALERT - Transaction {transaction_details.get('transaction_id')}"
            
            # Create email body
            body = self.create_email_body(transaction_details)
            message.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            server.sendmail(
                email_config['sender_email'],
                email_config['recipients'],
                message.as_string()
            )
            server.quit()
            
            logger.info(f"Email alert sent for transaction {transaction_details.get('transaction_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False
    
    def create_email_body(self, transaction_details):
        html_body = f"""
        <html>
        <body>
            <h2>🚨 FRAUD DETECTION ALERT</h2>
            <p>A potentially fraudulent transaction has been detected:</p>
            
            <table border="1" cellpadding="10" cellspacing="0">
                <tr>
                    <td><strong>Transaction ID:</strong></td>
                    <td>{transaction_details.get('transaction_id', 'N/A')}</td>
                </tr>
                <tr>
                    <td><strong>Amount:</strong></td>
                    <td>${transaction_details.get('amount', 0):.2f}</td>
                </tr>
                <tr>
                    <td><strong>Location:</strong></td>
                    <td>{transaction_details.get('location', 'N/A')}</td>
                </tr>
                <tr>
                    <td><strong>Fraud Probability:</strong></td>
                    <td>{transaction_details.get('fraud_probability', 0):.2%}</td>
                </tr>
                <tr>
                    <td><strong>Timestamp:</strong></td>
                    <td>{transaction_details.get('timestamp', 'N/A')}</td>
                </tr>
            </table>
            
            <p><strong>Immediate action required:</strong></p>
            <ul>
                <li>Review the transaction details</li>
                <li>Contact the customer if necessary</li>
                <li>Consider blocking the transaction or account</li>
            </ul>
            
            <p><em>This is an automated alert from the Fraud Detection System</em></p>
        </body>
        </html>
        """
        return html_body
    
    def send_slack_alert(self, transaction_details):
        if not self.config['slack']['enabled']:
            logger.info("Slack alerts are disabled")
            return False
        
        try:
            webhook_url = self.config['slack']['webhook_url']
            
            # Create Slack message
            slack_message = {
                "text": f"🚨 FRAUD ALERT - Transaction {transaction_details.get('transaction_id')}",
                "attachments": [
                    {
                        "color": "danger",
                        "fields": [
                            {
                                "title": "Transaction ID",
                                "value": str(transaction_details.get('transaction_id', 'N/A')),
                                "short": True
                            },
                            {
                                "title": "Amount",
                                "value": f"${transaction_details.get('amount', 0):.2f}",
                                "short": True
                            },
                            {
                                "title": "Location",
                                "value": transaction_details.get('location', 'N/A'),
                                "short": True
                            },
                            {
                                "title": "Fraud Probability",
                                "value": f"{transaction_details.get('fraud_probability', 0):.2%}",
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": transaction_details.get('timestamp', 'N/A'),
                                "short": False
                            }
                        ]
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(webhook_url, json=slack_message)
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent for transaction {transaction_details.get('transaction_id')}")
                return True
            else:
                logger.error(f"Slack API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False
    
    def send_alert(self, transaction_details):
        # Check if fraud probability exceeds threshold
        fraud_prob = transaction_details.get('fraud_probability', 0)
        threshold = self.config.get('threshold', 0.8)
        
        if fraud_prob < threshold:
            logger.info(f"Fraud probability {fraud_prob:.2%} below threshold {threshold:.2%}. No alert sent.")
            return False
        
        # Log the alert
        alert_record = {
            'transaction_id': transaction_details.get('transaction_id'),
            'timestamp': datetime.now().isoformat(),
            'fraud_probability': fraud_prob,
            'alert_type': 'email_slack'
        }
        self.alert_history.append(alert_record)
        
        # Send alerts through enabled channels
        email_sent = self.send_email_alert(transaction_details)
        slack_sent = self.send_slack_alert(transaction_details)
        
        return email_sent or slack_sent
    
    def get_alert_history(self, limit=100):
        return self.alert_history[-limit:]
    
    def save_alert_history(self, file_path='alert_history.json'):
        try:
            with open(file_path, 'w') as f:
                json.dump(self.alert_history, f, indent=2)
            logger.info(f"Alert history saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving alert history: {e}")
    
    def load_alert_history(self, file_path='alert_history.json'):
        try:
            with open(file_path, 'r') as f:
                self.alert_history = json.load(f)
            logger.info(f"Alert history loaded from {file_path}")
        except FileNotFoundError:
            logger.info(f"No existing alert history found at {file_path}")
        except Exception as e:
            logger.error(f"Error loading alert history: {e}")

# Standalone email function for quick use
def send_email_alert(transaction_details, sender_email, sender_password, recipients):
    try:
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = ', '.join(recipients)
        message['Subject'] = f"🚨 FRAUD ALERT - Transaction {transaction_details.get('transaction_id')}"
        
        body = f"""
        FRAUD DETECTION ALERT
        
        A potentially fraudulent transaction has been detected:
        
        Transaction ID: {transaction_details.get('transaction_id', 'N/A')}
        Amount: ${transaction_details.get('amount', 0):.2f}
        Location: {transaction_details.get('location', 'N/A')}
        Fraud Probability: {transaction_details.get('fraud_probability', 0):.2%}
        Timestamp: {transaction_details.get('timestamp', 'N/A')}
        
        Immediate action required:
        - Review the transaction details
        - Contact the customer if necessary
        - Consider blocking the transaction or account
        """
        
        message.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipients, message.as_string())
        server.quit()
        
        logger.info(f"Email alert sent for transaction {transaction_details.get('transaction_id')}")
        return True
        
    except Exception as e:
        logger.error(f"Error sending email alert: {e}")
        return False
