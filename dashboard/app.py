import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import joblib
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.supervised_models import SupervisedFraudDetector
from src.unsupervised_models import UnsupervisedFraudDetector

# Page configuration
st.set_page_config(
    page_title="Financial Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        animation: gradient 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .fraud-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .success-box {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stSelectbox>div>div>select {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = DataPreprocessor()
if 'supervised_detector' not in st.session_state:
    st.session_state.supervised_detector = SupervisedFraudDetector()
if 'unsupervised_detector' not in st.session_state:
    st.session_state.unsupervised_detector = UnsupervisedFraudDetector()

def load_data_from_db(db_path="fraud_detection.db"):
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM transactions", conn)
        conn.close()
        
        # Normalize column names to match expected format
        column_mapping = {
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
        
        # Apply column mapping if columns exist
        mapping_to_apply = {}
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                mapping_to_apply[old_col] = new_col
        
        if mapping_to_apply:
            df = df.rename(columns=mapping_to_apply)
        
        return df
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Financial Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🚀 Navigation")
    
    # Use pill buttons for navigation
    page = st.sidebar.selectbox(
        "Select Page",
        [
            "📊 Dashboard Overview",
            "📈 Model Performance", 
            "🔍 Transaction Analysis",
            "⚠️ Fraud Alerts",
            "⚙️ Model Training",
            "📥 Data Upload"
        ],
        index=0
    )
    
    # Dashboard Overview Page
    if page == "📊 Dashboard Overview":
        dashboard_overview()
    
    # Model Performance Page
    elif page == "📈 Model Performance":
        model_performance()
    
    # Transaction Analysis Page
    elif page == "🔍 Transaction Analysis":
        transaction_analysis()
    
    # Fraud Alerts Page
    elif page == "⚠️ Fraud Alerts":
        fraud_alerts()
    
    # Model Training Page
    elif page == "⚙️ Model Training":
        model_training()
    
    # Data Upload Page
    elif page == "📥 Data Upload":
        data_upload()

def dashboard_overview():
    st.header("📊 Dashboard Overview")
    
    # Load data
    df = load_data_from_db()
    
    if df is None or df.empty:
        st.warning("No data available. Please upload data first.")
        return
    
    st.session_state.data_loaded = True
    
    # Key Metrics with colorful cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(df)
    fraud_count = df['is_fraud'].sum() if 'is_fraud' in df.columns else df['Fraudulent'].sum() if 'Fraudulent' in df.columns else 0
    fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
    total_amount = df['amount'].sum() if 'amount' in df.columns and pd.api.types.is_numeric_dtype(df['amount']) else df['Transaction_Amount'].sum() if 'Transaction_Amount' in df.columns and pd.api.types.is_numeric_dtype(df['Transaction_Amount']) else 0
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size:1.2rem;">Total Transactions</h3>
            <p style="margin:0; font-size:2rem; font-weight:bold;">{total_transactions:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);">
            <h3 style="margin:0; font-size:1.2rem;">Fraudulent Transactions</h3>
            <p style="margin:0; font-size:2rem; font-weight:bold;">{fraud_count:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3 style="margin:0; font-size:1.2rem;">Fraud Rate</h3>
            <p style="margin:0; font-size:2rem; font-weight:bold;">{fraud_rate:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3 style="margin:0; font-size:1.2rem;">Total Amount</h3>
            <p style="margin:0; font-size:2rem; font-weight:bold;">${total_amount:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts with enhanced colors
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Fraud vs Legitimate Transactions")
        fraud_data = pd.DataFrame({
            'Type': ['Legitimate', 'Fraudulent'],
            'Count': [total_transactions - fraud_count, fraud_count]
        })
        fig_pie = px.pie(fraud_data, values='Count', names='Type', 
                        color_discrete_map={'Legitimate': '#2ecc71', 'Fraudulent': '#e74c3c'},
                        hole=0.4,
                        title='Transaction Distribution')
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        amount_col = 'amount' if 'amount' in df.columns else 'Transaction_Amount' if 'Transaction_Amount' in df.columns else None
        fraud_col = 'is_fraud' if 'is_fraud' in df.columns else 'Fraudulent' if 'Fraudulent' in df.columns else None
        
        if amount_col and fraud_col and pd.api.types.is_numeric_dtype(df[amount_col]):
            st.subheader("💰 Transaction Amount Distribution")
            fig_hist = px.histogram(df, x=amount_col, color=fraud_col,
                                   color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                                   nbins=50, title="Amount Distribution by Fraud Status",
                                   marginal='box')
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Amount distribution chart not available - missing required columns")
    
    # Time-based analysis with enhanced visualization
    timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'Transaction_Date' if 'Transaction_Date' in df.columns else None
    fraud_col = 'is_fraud' if 'is_fraud' in df.columns else 'Fraudulent' if 'Fraudulent' in df.columns else None
    
    if timestamp_col and fraud_col:
        st.subheader("📅 Transaction Timeline")
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%d-%m-%Y %H:%M')
        except:
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%m-%d-%Y %H:%M')
            except:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='mixed')
        df['date'] = df[timestamp_col].dt.date
        daily_transactions = df.groupby('date').size().reset_index(name='count')
        daily_fraud = df[df[fraud_col] == 1].groupby('date').size().reset_index(name='fraud_count')
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=daily_transactions['date'],
            y=daily_transactions['count'],
            mode='lines+markers',
            name='Total Transactions',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8)
        ))
        fig_time.add_trace(go.Scatter(
            x=daily_fraud['date'],
            y=daily_fraud['fraud_count'],
            mode='lines+markers',
            name='Fraudulent Transactions',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8)
        ))
        fig_time.update_layout(
            title='Transaction Volume Over Time',
            xaxis_title='Date',
            yaxis_title='Count',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14),
            hovermode='x unified'
        )
        st.plotly_chart(fig_time, use_container_width=True)

def model_performance():
    st.header("📈 Model Performance")
    
    # Load model evaluation results
    try:
        supervised_results = joblib.load('models/supervised_results.pkl')
        st.session_state.model_trained = True
    except:
        st.warning("Model evaluation results not found. Please train models first in the Model Training section.")
        return
    
    try:
        unsupervised_results = joblib.load('models/unsupervised_results.pkl')
    except:
        unsupervised_results = None
    
    # Supervised Model Performance
    st.subheader("🤖 Supervised Learning Models")
    
    model_names = list(supervised_results.keys())
    auc_scores = [supervised_results[model]['auc_score'] for model in model_names]
    
    # Colorful bar chart
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
    fig_auc = px.bar(x=model_names, y=auc_scores, 
                     title='🏆 Model AUC Scores',
                     labels={'x': 'Model', 'y': 'AUC Score'},
                     color=model_names,
                     color_discrete_sequence=colors,
                     text=auc_scores,
                     text_auto='.3f')
    fig_auc.update_traces(textposition='outside')
    fig_auc.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        showlegend=False
    )
    st.plotly_chart(fig_auc, use_container_width=True)
    
    # Detailed metrics for each model with tabs
    selected_model = st.selectbox("🎯 Select Model for Detailed Metrics", model_names)
    
    if selected_model:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 {selected_model} - Classification Report")
            report_df = pd.DataFrame(supervised_results[selected_model]['classification_report']).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', axis=0))
        
        with col2:
            st.subheader(f"🎯 {selected_model} - Confusion Matrix")
            cm = supervised_results[selected_model]['confusion_matrix']
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                              color_continuous_scale='RdYlGn',
                              title='Confusion Matrix',
                              labels=dict(x="Predicted", y="Actual", color="Count"))
            fig_cm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
    
    # Unsupervised Model Performance
    st.subheader("🔍 Unsupervised Learning Models")
    
    if unsupervised_results:
        unsupervised_names = list(unsupervised_results.keys())
        unsupervised_auc = [unsupervised_results[model]['auc_score'] for model in unsupervised_names]
        
        # Colorful bar chart for unsupervised
        unsupervised_colors = ['#ff6b6b', '#feca57', '#48dbfb']
        fig_unsupervised = px.bar(x=unsupervised_names, y=unsupervised_auc,
                                 title='🔍 Unsupervised Model AUC Scores',
                                 labels={'x': 'Model', 'y': 'AUC Score'},
                                 color=unsupervised_names,
                                 color_discrete_sequence=unsupervised_colors,
                                 text=unsupervised_auc,
                                 text_auto='.3f')
        fig_unsupervised.update_traces(textposition='outside')
        fig_unsupervised.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14),
            showlegend=False
        )
        st.plotly_chart(fig_unsupervised, use_container_width=True)
    else:
        st.info("No unsupervised models trained yet.")

def transaction_analysis():
    st.header("🔍 Transaction Analysis")
    
    df = load_data_from_db()
    
    if df is None or df.empty:
        st.warning("No data available. Please upload data first.")
        return
    
    # Interactive filters with colorful design
    st.subheader("🎛️ Filter Controls")
    
    col1, col2, col3 = st.columns(3)
    
    # Ensure amount column exists and is numeric
    amount_col = 'amount' if 'amount' in df.columns else 'Transaction_Amount' if 'Transaction_Amount' in df.columns else None
    if amount_col and pd.api.types.is_numeric_dtype(df[amount_col]):
        max_amount_value = float(df[amount_col].max())
        min_amount_value = 0.0
    else:
        max_amount_value = 1000.0
        min_amount_value = 0.0
    
    with col1:
        min_amount = st.slider("💰 Minimum Amount", min_value=min_amount_value, max_value=max_amount_value, value=min_amount_value, step=10.0)
    
    with col2:
        max_amount = st.slider("💰 Maximum Amount", min_value=min_amount_value, max_value=max_amount_value, value=max_amount_value, step=10.0)
    
    with col3:
        fraud_filter = st.selectbox("🚨 Filter by Fraud Status", ["All", "Legitimate", "Fraudulent"])
    
    # Apply filters - only if amount column exists
    if amount_col and pd.api.types.is_numeric_dtype(df[amount_col]):
        filtered_df = df[(df[amount_col] >= min_amount) & (df[amount_col] <= max_amount)]
    else:
        filtered_df = df.copy()
    
    fraud_col = 'is_fraud' if 'is_fraud' in df.columns else 'Fraudulent' if 'Fraudulent' in df.columns else None
    if fraud_col:
        if fraud_filter == "Legitimate":
            filtered_df = filtered_df[filtered_df[fraud_col] == 0]
        elif fraud_filter == "Fraudulent":
            filtered_df = filtered_df[filtered_df[fraud_col] == 1]
    
    # Display results with colored indicators
    st.subheader(f"📋 Filtered Transactions ({len(filtered_df)} records)")
    
    if len(filtered_df) > 0:
        # Add color-coded fraud status
        def color_fraud_status(val):
            color = '#ff6b6b' if val == 1 else '#2ecc71'
            return f'background-color: {color}'
        
        fraud_col = 'is_fraud' if 'is_fraud' in filtered_df.columns else 'Fraudulent' if 'Fraudulent' in filtered_df.columns else None
        if fraud_col:
            styled_df = filtered_df.head(100).style.map(color_fraud_status, subset=[fraud_col])
            st.dataframe(styled_df)
        else:
            st.dataframe(filtered_df.head(100))
    else:
        st.info("No transactions match the current filters.")
    
    # Location analysis with enhanced visualization
    location_col = 'location' if 'location' in df.columns else 'Location' if 'Location' in df.columns else None
    if location_col:
        st.subheader("🌍 Transactions by Location")
        location_counts = df[location_col].value_counts().head(20)
        
        if len(location_counts) > 0:
            # Colorful bar chart
            colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3']
            fig_location = px.bar(x=location_counts.index, y=location_counts.values,
                                title='Top 20 Locations by Transaction Count',
                                labels={'x': 'Location', 'y': 'Transaction Count'},
                                color=location_counts.values,
                                color_continuous_scale='Viridis',
                                text_auto=True)
            fig_location.update_traces(textposition='outside')
            fig_location.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14),
                showlegend=False
            )
            st.plotly_chart(fig_location, use_container_width=True)
        else:
            st.info("No location data available")
    
    # Device analysis with enhanced visualization
    device_col = 'device' if 'device' in df.columns else 'Device_Type' if 'Device_Type' in df.columns else None
    if device_col:
        st.subheader("📱 Transactions by Device")
        device_counts = df[device_col].value_counts()
        
        if len(device_counts) > 0:
            # Colorful pie chart
            device_colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#ff6b6b']
            fig_device = px.pie(values=device_counts.values, names=device_counts.index,
                               title='Transaction Distribution by Device',
                               hole=0.4,
                               color_discrete_sequence=device_colors)
            fig_device.update_traces(textposition='inside', textinfo='percent+label')
            fig_device.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14)
            )
            st.plotly_chart(fig_device, use_container_width=True)
        else:
            st.info("No device data available")
    
    # Interactive amount range visualization - only if amount column exists
    if amount_col and fraud_col and pd.api.types.is_numeric_dtype(df[amount_col]):
        st.subheader("💵 Amount Range Analysis")
        fig_amount = px.box(df, x=fraud_col, y=amount_col,
                           color=fraud_col,
                           color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                           title='Amount Distribution by Fraud Status',
                           labels={fraud_col: 'Fraud Status', amount_col: 'Transaction Amount'})
        fig_amount.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14)
        )
        st.plotly_chart(fig_amount, use_container_width=True)

def fraud_alerts():
    st.header("⚠️ Fraud Alerts")
    
    try:
        conn = sqlite3.connect("fraud_detection.db")
        alerts_df = pd.read_sql("SELECT * FROM fraud_alerts ORDER BY alert_timestamp DESC LIMIT 100", conn)
        conn.close()
    except Exception as e:
        st.warning(f"No fraud alerts found in database. Error: {e}")
        st.info("Fraud alerts are generated by running the real-time monitoring system locally.")
        st.info("Run: python run_real_time_alerts.py")
        return
    
    if alerts_df.empty:
        st.info("No recent fraud alerts.")
        return
    
    # Alert statistics with colorful cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h3 style="margin:0; font-size:1.2rem;">Total Alerts</h3>
            <p style="margin:0; font-size:2rem; font-weight:bold;">{len(alerts_df)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_risk = (alerts_df['fraud_probability'] > 0.9).sum() if 'fraud_probability' in alerts_df.columns else 0
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);">
            <h3 style="margin:0; font-size:1.2rem;">High Risk Alerts</h3>
            <p style="margin:0; font-size:2rem; font-weight:bold;">{high_risk}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_prob = alerts_df['fraud_probability'].mean() if 'fraud_probability' in alerts_df.columns else 0
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3 style="margin:0; font-size:1.2rem;">Avg Fraud Probability</h3>
            <p style="margin:0; font-size:2rem; font-weight:bold;">{avg_prob:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent alerts with enhanced display
    st.subheader("🚨 Recent Fraud Alerts")
    
    for idx, row in alerts_df.head(10).iterrows():
        risk_level = "🔴 HIGH" if row['fraud_probability'] > 0.9 else "🟡 MEDIUM" if row['fraud_probability'] > 0.7 else "🟢 LOW"
        risk_color = "#ff6b6b" if row['fraud_probability'] > 0.9 else "#feca57" if row['fraud_probability'] > 0.7 else "#2ecc71"
        
        timestamp_display = row.get('alert_timestamp', row.get('timestamp', 'N/A'))
        with st.expander(f"Alert {row['transaction_id']} - {risk_level} - {timestamp_display}"):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {risk_color}22 0%, {risk_color}44 100%); 
                        padding: 1rem; border-radius: 10px; border-left: 4px solid {risk_color};">
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💰 Amount", f"${row.get('amount', 0):.2f}")
            with col2:
                st.metric("📍 Location", row.get('location', 'N/A'))
            with col3:
                st.metric("🎯 Fraud Probability", f"{row.get('fraud_probability', 0):.2%}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Alert timeline with enhanced visualization
    st.subheader("📊 Alert Timeline")
    if 'alert_timestamp' in alerts_df.columns:
        alerts_df['alert_timestamp'] = pd.to_datetime(alerts_df['alert_timestamp'])
        alerts_by_hour = alerts_df.groupby(alerts_df['alert_timestamp'].dt.hour).size()
    else:
        st.info("Timeline not available - alert_timestamp column missing")
        alerts_by_hour = pd.Series()
    
    if len(alerts_by_hour) > 0:
        fig_timeline = px.line(x=alerts_by_hour.index, y=alerts_by_hour.values,
                             title='Alerts by Hour of Day',
                             labels={'x': 'Hour', 'y': 'Number of Alerts'},
                             line_shape='spline',
                             markers=True)
        fig_timeline.update_traces(line=dict(color='#667eea', width=3), marker=dict(size=8))
        fig_timeline.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14)
        )
        fig_timeline.add_hrect(y0=0, y1=alerts_by_hour.max(), 
                              fillcolor="rgba(102, 126, 234, 0.1)", 
                              layer="below", line_width=0)
        st.plotly_chart(fig_timeline, use_container_width=True)

def model_training():
    st.header("⚙️ Model Training")
    
    df = load_data_from_db()
    
    if df is None or df.empty:
        st.warning("No data available. Please upload data first.")
        return
    
    # Preprocessing options with enhanced UI
    st.subheader("🔧 Data Preprocessing")
    
    # Get available categorical columns
    available_categorical = df.select_dtypes(include=['object']).columns.tolist()
    
    # Set default values only for columns that actually exist
    default_categorical = []
    for col in ['location', 'device', 'merchant_category', 'payment_method']:
        if col in available_categorical:
            default_categorical.append(col)
    
    categorical_columns = st.multiselect(
        "🏷️ Select Categorical Columns",
        available_categorical,
        default=default_categorical
    )
    
    # Get available numeric columns
    available_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Set default values only for columns that actually exist
    default_numeric = []
    for col in ['amount', 'is_international', 'previous_transactions', 'average_spend', 'account_age_days', 'suspicious_keyword']:
        if col in available_numeric:
            default_numeric.append(col)
    
    numeric_columns = st.multiselect(
        "🔢 Select Numeric Columns for Normalization",
        available_numeric,
        default=default_numeric
    )
    
    if st.button("🚀 Preprocess Data", key="preprocess"):
        with st.spinner("Preprocessing data..."):
            try:
                # Handle missing values
                df_cleaned = st.session_state.preprocessor.handle_missing_values(df)
                
                # Encode categorical features
                if categorical_columns:
                    df_cleaned = st.session_state.preprocessor.encode_categorical_features(
                        df_cleaned, categorical_columns
                    )
                
                # Feature engineering
                df_cleaned = st.session_state.preprocessor.feature_engineering(df_cleaned)
                
                # Normalize features
                if numeric_columns:
                    df_cleaned = st.session_state.preprocessor.normalize_features(
                        df_cleaned, numeric_columns
                    )
                
                # Additional NaN handling - ensure no NaN values remain
                df_cleaned = df_cleaned.fillna(0)
                
                # Check for any remaining NaN values
                if df_cleaned.isnull().any().any():
                    st.warning("⚠️ Some NaN values still present, filling with 0")
                    df_cleaned = df_cleaned.fillna(0)
                
                # Prepare training data
                X_train, X_test, y_train, y_test = st.session_state.preprocessor.prepare_training_data(
                    df_cleaned
                )
                
                # Final check for NaN in training data
                if X_train.isnull().any().any():
                    st.warning("⚠️ NaN values found in training data, filling with 0")
                    X_train = X_train.fillna(0)
                    X_test = X_test.fillna(0)
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                st.markdown("""
                <div class="success-box">
                    <h3>✅ Data preprocessing completed!</h3>
                </div>
                """, unsafe_allow_html=True)
                st.info(f"📊 Training set: {X_train.shape}, Test set: {X_test.shape}")
                st.info(f"✅ No NaN values in training data: {not X_train.isnull().any().any()}")
                
            except Exception as e:
                st.error(f"❌ Error during preprocessing: {e}")
    
    # Model training options with enhanced UI
    st.subheader("🤖 Model Training")
    
    model_type = st.selectbox("🎯 Select Model Type", ["Supervised Learning", "Unsupervised Learning"])
    
    if model_type == "Supervised Learning":
        supervised_models = st.multiselect(
            "📈 Select Models to Train",
            ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"],
            default=["Logistic Regression", "Random Forest"]
        )
        
        if st.button("🚀 Train Supervised Models", key="train_supervised") and hasattr(st.session_state, 'X_train'):
            with st.spinner("Training supervised models..."):
                try:
                    progress_bar = st.progress(0)
                    total_models = len([m for m in supervised_models if m in ["Logistic Regression", "Random Forest"]])
                    current_model = 0
                    
                    if "Logistic Regression" in supervised_models:
                        st.session_state.supervised_detector.train_logistic_regression(
                            st.session_state.X_train, st.session_state.y_train
                        )
                        current_model += 1
                        progress_bar.progress(current_model / total_models)
                    
                    if "Random Forest" in supervised_models:
                        st.session_state.supervised_detector.train_random_forest(
                            st.session_state.X_train, st.session_state.y_train
                        )
                        current_model += 1
                        progress_bar.progress(current_model / total_models)
                    
                    if "XGBoost" in supervised_models:
                        st.session_state.supervised_detector.train_xgboost(
                            st.session_state.X_train, st.session_state.y_train
                        )
                    
                    if "LightGBM" in supervised_models:
                        st.session_state.supervised_detector.train_lightgbm(
                            st.session_state.X_train, st.session_state.y_train
                        )
                    
                    # Evaluate models
                    results = st.session_state.supervised_detector.evaluate_all_models(
                        st.session_state.X_test, st.session_state.y_test
                    )
                    
                    joblib.dump(results, 'models/supervised_results.pkl')
                    st.session_state.supervised_detector.save_models()
                    
                    st.session_state.model_trained = True
                    
                    st.markdown("""
                    <div class="success-box">
                        <h3>✅ Supervised models trained and saved successfully!</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error training models: {e}")
    
    else:
        unsupervised_models = st.multiselect(
            "🔍 Select Models to Train",
            ["Isolation Forest", "One-Class SVM", "Autoencoder"],
            default=["Isolation Forest"]
        )
        
        if st.button("🚀 Train Unsupervised Models", key="train_unsupervised") and hasattr(st.session_state, 'X_train'):
            with st.spinner("Training unsupervised models..."):
                try:
                    if "Isolation Forest" in unsupervised_models:
                        st.session_state.unsupervised_detector.train_isolation_forest(
                            st.session_state.X_train
                        )
                    
                    if "One-Class SVM" in unsupervised_models:
                        st.session_state.unsupervised_detector.train_one_class_svm(
                            st.session_state.X_train
                        )
                    
                    if "Autoencoder" in unsupervised_models:
                        st.session_state.unsupervised_detector.train_autoencoder(
                            st.session_state.X_train
                        )
                    
                    # Evaluate models
                    results = st.session_state.unsupervised_detector.evaluate_all_models(
                        st.session_state.X_test, st.session_state.y_test
                    )
                    
                    joblib.dump(results, 'models/unsupervised_results.pkl')
                    st.session_state.unsupervised_detector.save_models()
                    
                    st.session_state.model_trained = True
                    
                    st.markdown("""
                    <div class="success-box">
                        <h3>✅ Unsupervised models trained and saved successfully!</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error training models: {e}")

def data_upload():
    st.header("📥 Data Upload")
    
    upload_method = st.selectbox("🎯 Select Upload Method", ["Upload File", "Generate Sample Data"])
    
    if upload_method == "Upload File":
        uploaded_file = st.file_uploader(
            "📁 Choose a CSV or JSON file",
            type=['csv', 'json']
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_json(uploaded_file)
                
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ File uploaded successfully!</h3>
                    <p>Shape: {df.shape}</p>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(df.head())
                
                if st.button("🚀 Process and Save to Database", key="process_upload"):
                    with st.spinner("Processing data..."):
                        success = st.session_state.preprocessor.etl_process(df)
                        if success:
                            st.markdown("""
                            <div class="success-box">
                                <h3>✅ Data processed and saved to database successfully!</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="fraud-alert">
                                <h3>❌ Error processing data</h3>
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h3>❌ Error reading file: {e}</h3>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.subheader("🎲 Generate Sample Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_samples = st.slider("📊 Number of Samples", min_value=1000, max_value=50000, value=10000, step=1000)
        
        with col2:
            fraud_ratio = st.slider("🎯 Fraud Ratio", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        
        if st.button("🚀 Generate and Process Data", key="generate_data"):
            with st.spinner("Generating and processing data..."):
                try:
                    from src.generate_sample_data import generate_fraud_data
                    df = generate_fraud_data(num_samples, fraud_ratio)
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ Sample data generated successfully!</h3>
                        <p>Shape: {df.shape}</p>
                        <p>Fraud Rate: {fraud_ratio:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(df.head())
                    
                    success = st.session_state.preprocessor.etl_process(df)
                    if success:
                        st.markdown("""
                        <div class="success-box">
                            <h3>✅ Data processed and saved to database successfully!</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="fraud-alert">
                            <h3>❌ Error processing data</h3>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                    <div class="fraud-alert">
                        <h3>❌ Error generating data: {e}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    from src.generate_sample_data import generate_fraud_data
                    
                    df = generate_fraud_data(num_samples, fraud_ratio)
                    st.success(f"Generated {len(df)} samples with {df['is_fraud'].sum()} fraudulent transactions")
                    st.dataframe(df.head())
                    
                    if st.button("Save Generated Data to Database"):
                        success = st.session_state.preprocessor.etl_process(df)
                        if success:
                            st.success("Sample data saved to database successfully!")
                        else:
                            st.error("Error saving data.")
                            
                except Exception as e:
                    st.error(f"Error generating data: {e}")

if __name__ == "__main__":
    main()
