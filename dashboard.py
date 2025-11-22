"""
Advanced Analytics Dashboard for Insurance Fraud Detection
Features:
- Time-series analysis of fraud trends
- Geographic fraud patterns
- Risk factor analysis
- Interactive visualizations
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Insurance Fraud Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess data from multiple files"""
    try:
        # Try to load File01 (has labels)
        try:
            df1 = pd.read_csv("data/Auto_Insurance_Fraud_Claims_File01.csv")
        except:
            df1 = pd.read_csv("Auto_Insurance_Fraud_Claims_File01.csv")
        df1['source'] = 'File01'
        
        # Load File002
        try:
            df2 = pd.read_csv("data/Auto_Insurance_Fraud_Claims_File002.csv")
        except:
            df2 = pd.read_csv("Auto_Insurance_Fraud_Claims_File002.csv")
        df2['source'] = 'File002'
        
        # Combine datasets
        df = pd.concat([df1, df2], ignore_index=True)
        
        # Convert date columns
        date_cols = ['Claims_Date', 'Accident_Date', 'Policy_Start_Date', 'Policy_Expiry_Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Create fraud indicator (1 for fraud, 0 for not fraud)
        if 'Fraud_Ind' in df.columns:
            df['Fraud'] = df['Fraud_Ind'].map({'Y': 1, 'N': 0})
        else:
            df['Fraud'] = 0
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def load_model():
    """Load the fraud detection model"""
    try:
        model = joblib.load("model/best_fraud_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_time_series_analysis(df):
    """Create time-series analysis of fraud trends"""
    if 'Claims_Date' not in df.columns or df['Claims_Date'].isna().all():
        return None
    
    # Group by month
    df['Year_Month'] = df['Claims_Date'].dt.to_period('M').astype(str)
    monthly_fraud = df.groupby('Year_Month').agg({
        'Fraud': ['sum', 'count', 'mean']
    }).reset_index()
    monthly_fraud.columns = ['Year_Month', 'Fraud_Count', 'Total_Claims', 'Fraud_Rate']
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Fraud Cases Over Time', 'Fraud Rate Over Time'),
        vertical_spacing=0.15
    )
    
    # Fraud count
    fig.add_trace(
        go.Scatter(
            x=monthly_fraud['Year_Month'],
            y=monthly_fraud['Fraud_Count'],
            mode='lines+markers',
            name='Fraud Cases',
            line=dict(color='#d62728', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Fraud rate
    fig.add_trace(
        go.Scatter(
            x=monthly_fraud['Year_Month'],
            y=monthly_fraud['Fraud_Rate'] * 100,
            mode='lines+markers',
            name='Fraud Rate (%)',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=8),
            fill='tonexty'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Number of Cases", row=1, col=1)
    fig.update_yaxes(title_text="Fraud Rate (%)", row=2, col=1)
    fig.update_layout(height=600, showlegend=True, title_text="Time-Series Fraud Analysis")
    
    return fig, monthly_fraud

def create_geographic_analysis(df):
    """Create geographic fraud analysis"""
    if 'Acccident_State' not in df.columns:
        return None
    
    state_fraud = df.groupby('Acccident_State').agg({
        'Fraud': ['sum', 'count', 'mean']
    }).reset_index()
    state_fraud.columns = ['State', 'Fraud_Count', 'Total_Claims', 'Fraud_Rate']
    state_fraud = state_fraud.sort_values('Fraud_Count', ascending=False).head(15)
    
    fig = px.bar(
        state_fraud,
        x='State',
        y=['Fraud_Count', 'Total_Claims'],
        title="Top 15 States by Fraud Cases",
        labels={'value': 'Number of Cases', 'State': 'State'},
        barmode='group',
        color_discrete_map={'Fraud_Count': '#d62728', 'Total_Claims': '#2ca02c'}
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    
    return fig, state_fraud

def create_risk_factor_analysis(df):
    """Analyze risk factors for fraud"""
    risk_factors = {}
    
    # Age analysis
    if 'Age_Insured' in df.columns:
        age_bins = [0, 25, 35, 45, 55, 65, 100]
        age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        df['Age_Group'] = pd.cut(df['Age_Insured'], bins=age_bins, labels=age_labels)
        age_fraud = df.groupby('Age_Group')['Fraud'].mean().reset_index()
        risk_factors['Age'] = age_fraud
    
    # Accident Severity
    if 'Accident_Severity' in df.columns:
        severity_fraud = df.groupby('Accident_Severity')['Fraud'].mean().reset_index()
        risk_factors['Accident_Severity'] = severity_fraud
    
    # Vehicle Cost
    if 'Vehicle_Cost' in df.columns:
        cost_bins = [0, 10000, 20000, 30000, 50000, float('inf')]
        cost_labels = ['<10k', '10k-20k', '20k-30k', '30k-50k', '50k+']
        df['Cost_Group'] = pd.cut(df['Vehicle_Cost'], bins=cost_bins, labels=cost_labels)
        cost_fraud = df.groupby('Cost_Group')['Fraud'].mean().reset_index()
        risk_factors['Vehicle_Cost'] = cost_fraud
    
    # Total Claim Amount
    if 'Total_Claim' in df.columns:
        claim_bins = [0, 5000, 10000, 20000, 50000, float('inf')]
        claim_labels = ['<5k', '5k-10k', '10k-20k', '20k-50k', '50k+']
        df['Claim_Group'] = pd.cut(df['Total_Claim'], bins=claim_bins, labels=claim_labels)
        claim_fraud = df.groupby('Claim_Group')['Fraud'].mean().reset_index()
        risk_factors['Total_Claim'] = claim_fraud
    
    return risk_factors, df

def create_risk_factor_visualizations(risk_factors):
    """Create visualizations for risk factors"""
    figs = []
    
    if 'Age' in risk_factors:
        fig = px.bar(
            risk_factors['Age'],
            x='Age_Group',
            y='Fraud',
            title="Fraud Rate by Age Group",
            labels={'Fraud': 'Fraud Rate', 'Age_Group': 'Age Group'},
            color='Fraud',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        figs.append(('Age Analysis', fig))
    
    if 'Accident_Severity' in risk_factors:
        fig = px.bar(
            risk_factors['Accident_Severity'],
            x='Accident_Severity',
            y='Fraud',
            title="Fraud Rate by Accident Severity",
            labels={'Fraud': 'Fraud Rate', 'Accident_Severity': 'Severity'},
            color='Fraud',
            color_continuous_scale='Oranges'
        )
        fig.update_layout(height=400)
        figs.append(('Severity Analysis', fig))
    
    if 'Vehicle_Cost' in risk_factors:
        fig = px.bar(
            risk_factors['Vehicle_Cost'],
            x='Cost_Group',
            y='Fraud',
            title="Fraud Rate by Vehicle Cost",
            labels={'Fraud': 'Fraud Rate', 'Cost_Group': 'Vehicle Cost'},
            color='Fraud',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        figs.append(('Vehicle Cost Analysis', fig))
    
    if 'Total_Claim' in risk_factors:
        fig = px.bar(
            risk_factors['Total_Claim'],
            x='Claim_Group',
            y='Fraud',
            title="Fraud Rate by Claim Amount",
            labels={'Fraud': 'Fraud Rate', 'Claim_Group': 'Claim Amount'},
            color='Fraud',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=400)
        figs.append(('Claim Amount Analysis', fig))
    
    return figs

def main():
    st.markdown('<h1 class="main-header">📊 Insurance Fraud Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Unable to load data. Please check data files.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    if 'Claims_Date' in df.columns and not df['Claims_Date'].isna().all():
        min_date = df['Claims_Date'].min()
        max_date = df['Claims_Date'].max()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        if len(date_range) == 2:
            df = df[(df['Claims_Date'] >= pd.Timestamp(date_range[0])) & 
                   (df['Claims_Date'] <= pd.Timestamp(date_range[1]))]
    
    # State filter
    if 'Acccident_State' in df.columns:
        states = ['All'] + sorted(df['Acccident_State'].dropna().unique().tolist())
        selected_state = st.sidebar.selectbox("State", states)
        if selected_state != 'All':
            df = df[df['Acccident_State'] == selected_state]
    
    # Key Metrics
    st.header("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_claims = len(df)
    fraud_count = df['Fraud'].sum() if 'Fraud' in df.columns else 0
    fraud_rate = (fraud_count / total_claims * 100) if total_claims > 0 else 0
    avg_claim = df['Total_Claim'].mean() if 'Total_Claim' in df.columns else 0
    
    col1.metric("Total Claims", f"{total_claims:,}")
    col2.metric("Fraud Cases", f"{fraud_count:,}", f"{fraud_rate:.2f}%")
    col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    col4.metric("Avg Claim Amount", f"${avg_claim:,.2f}")
    
    # Time Series Analysis
    st.header("📅 Time-Series Analysis")
    if 'Claims_Date' in df.columns and not df['Claims_Date'].isna().all():
        time_fig, monthly_data = create_time_series_analysis(df)
        if time_fig:
            st.plotly_chart(time_fig, use_container_width=True)
            with st.expander("View Monthly Data"):
                st.dataframe(monthly_data)
    else:
        st.info("Date information not available for time-series analysis.")
    
    # Geographic Analysis
    st.header("🗺️ Geographic Analysis")
    if 'Acccident_State' in df.columns:
        geo_fig, state_data = create_geographic_analysis(df)
        if geo_fig:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(geo_fig, use_container_width=True)
            with col2:
                st.dataframe(state_data)
    else:
        st.info("Geographic data not available.")
    
    # Risk Factor Analysis
    st.header("⚠️ Risk Factor Analysis")
    risk_factors, df_processed = create_risk_factor_analysis(df)
    risk_viz = create_risk_factor_visualizations(risk_factors)
    
    if risk_viz:
        for title, fig in risk_viz:
            st.subheader(title)
            st.plotly_chart(fig, use_container_width=True)
    
    # Additional Insights
    st.header("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Fraud Indicators")
        insights = []
        
        if 'Accident_Severity' in df.columns:
            severity_insight = df.groupby('Accident_Severity')['Fraud'].mean().sort_values(ascending=False)
            if len(severity_insight) > 0:
                top_severity = severity_insight.index[0]
                insights.append(f"🔴 Highest fraud rate: {top_severity} accidents ({severity_insight[top_severity]*100:.1f}%)")
        
        if 'Age_Insured' in df.columns and 'Age_Group' in df_processed.columns:
            age_insight = df_processed.groupby('Age_Group')['Fraud'].mean().sort_values(ascending=False)
            if len(age_insight) > 0:
                top_age = age_insight.index[0]
                insights.append(f"👤 Highest risk age group: {top_age} ({age_insight[top_age]*100:.1f}%)")
        
        for insight in insights[:5]:
            st.write(insight)
    
    with col2:
        st.subheader("Fraud Patterns")
        patterns = []
        
        if 'Total_Claim' in df.columns:
            high_claim_fraud = df[df['Total_Claim'] > df['Total_Claim'].quantile(0.75)]['Fraud'].mean()
            patterns.append(f"💰 High-value claims (>75th percentile): {high_claim_fraud*100:.1f}% fraud rate")
        
        if 'Vehicle_Cost' in df.columns:
            low_cost_fraud = df[df['Vehicle_Cost'] < df['Vehicle_Cost'].quantile(0.25)]['Fraud'].mean()
            patterns.append(f"🚗 Low-value vehicles (<25th percentile): {low_cost_fraud*100:.1f}% fraud rate")
        
        for pattern in patterns[:5]:
            st.write(pattern)
    
    # Footer
    st.markdown("---")
    st.markdown("**Dashboard created with Streamlit, Plotly, and Machine Learning**")

if __name__ == "__main__":
    main()

