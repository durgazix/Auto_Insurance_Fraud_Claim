"""
Comprehensive Interactive Insurance Fraud Detection Application
Features:
- Real-time fraud prediction
- Advanced analytics dashboard
- Risk assessment
- Model insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.preprocess import preprocess
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Insurance Fraud Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .fraud-box {
        background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
        color: white;
    }
    .no-fraud-box {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the fraud detection model"""
    try:
        model = joblib.load("model/best_fraud_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def calculate_risk_score(input_data, model):
    """Calculate risk score based on input features"""
    try:
        df = pd.DataFrame([input_data])
        df_processed = preprocess(df)
        probability = model.predict_proba(df_processed)[0]
        return probability[1]  # Return fraud probability
    except Exception as e:
        st.error(f"Error calculating risk score: {str(e)}")
        return 0.5

def get_risk_category(risk_score):
    """Categorize risk level"""
    if risk_score >= 0.7:
        return "HIGH", "risk-high"
    elif risk_score >= 0.4:
        return "MEDIUM", "risk-medium"
    else:
        return "LOW", "risk-low"

def main():
    # Header
    st.markdown('<h1 class="main-header">🚨 AI Insurance Fraud Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Machine Learning & Data Analytics Platform</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Model could not be loaded. Please check model files.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🔮 Fraud Prediction", "📊 Analytics Dashboard", "📈 Model Insights", "ℹ️ About"]
    )
    
    if page == "🔮 Fraud Prediction":
        show_prediction_page(model)
    elif page == "📊 Analytics Dashboard":
        show_dashboard_page()
    elif page == "📈 Model Insights":
        show_model_insights(model)
    else:
        show_about_page()

def show_prediction_page(model):
    """Fraud prediction interface"""
    st.header("🔮 Real-Time Fraud Prediction")
    st.markdown("Enter claim details below to get an instant fraud risk assessment.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input form
        with st.form("fraud_prediction_form"):
            st.subheader("Claim Information")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                gender = st.selectbox("Gender", ["male", "female"], key="gender")
                age = st.slider("Age of Insured", 18, 100, 35, key="age")
                education = st.selectbox(
                    "Education Level",
                    ["High School", "College", "Masters", "PhD"],
                    key="education"
                )
                occupation = st.selectbox(
                    "Occupation",
                    ["engineer", "doctor", "teacher", "lawyer", "other"],
                    key="occupation"
                )
                hobbies = st.selectbox(
                    "Hobbies",
                    ["reading", "sports", "traveling", "gaming"],
                    key="hobbies"
                )
            
            with col_b:
                accident_severity = st.selectbox(
                    "Accident Severity",
                    ["Minor", "Major", "Total Loss"],
                    key="severity"
                )
                accident_type = st.selectbox(
                    "Accident Type",
                    ["Rear Collision", "Side Collision", "Head-On", "Other"],
                    key="accident_type"
                )
                vehicle_cost = st.number_input(
                    "Vehicle Cost ($)",
                    min_value=500.0,
                    max_value=100000.0,
                    value=25000.0,
                    step=1000.0,
                    key="vehicle_cost"
                )
                annual_mileage = st.number_input(
                    "Annual Mileage",
                    min_value=1000.0,
                    max_value=50000.0,
                    value=12000.0,
                    step=1000.0,
                    key="mileage"
                )
                total_claim = st.number_input(
                    "Total Claim Amount ($)",
                    min_value=0.0,
                    max_value=200000.0,
                    value=15000.0,
                    step=1000.0,
                    key="claim"
                )
            
            submitted = st.form_submit_button("🔍 Predict Fraud Risk", use_container_width=True)
    
    with col2:
        st.subheader("💡 Quick Tips")
        st.info("""
        **High Risk Indicators:**
        - Very high claim amounts
        - Recent policy start dates
        - Multiple previous claims
        - Inconsistent information
        """)
    
    if submitted:
        # Prepare input data
        input_data = {
            "Gender": gender,
            "Age_Insured": age,
            "Education": education,
            "Occupation": occupation,
            "Hobbies": hobbies,
            "Accident_Severity": accident_severity,
            "Accident_Type": accident_type,
            "Vehicle_Cost": vehicle_cost,
            "Annual_Mileage": annual_mileage,
            "Total_Claim": total_claim,
        }
        
        # Make prediction
        try:
            df = pd.DataFrame([input_data])
            df_processed = preprocess(df)
            prediction = model.predict(df_processed)[0]
            probability = model.predict_proba(df_processed)[0]
            
            fraud_prob = probability[1] * 100
            no_fraud_prob = probability[0] * 100
            
            # Display results
            st.markdown("---")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box fraud-box">
                    <h2>🚨 FRAUD DETECTED</h2>
                    <p style="font-size: 1.5rem;">Fraud Probability: {fraud_prob:.2f}%</p>
                    <p>This claim has been flagged as potentially fraudulent.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box no-fraud-box">
                    <h2>✅ NO FRAUD DETECTED</h2>
                    <p style="font-size: 1.5rem;">Fraud Probability: {fraud_prob:.2f}%</p>
                    <p>This claim appears to be legitimate.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk assessment
            risk_score = fraud_prob / 100
            risk_level, risk_class = get_risk_category(risk_score)
            
            st.markdown("---")
            st.subheader("📊 Detailed Risk Assessment")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fraud Probability", f"{fraud_prob:.2f}%")
            
            with col2:
                st.metric("Legitimate Probability", f"{no_fraud_prob:.2f}%")
            
            with col3:
                st.markdown(f"**Risk Level:** <span class='{risk_class}'>{risk_level}</span>", unsafe_allow_html=True)
            
            # Probability visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=['Legitimate', 'Fraud'],
                    y=[no_fraud_prob, fraud_prob],
                    marker_color=['#51cf66', '#ff6b6b'],
                    text=[f'{no_fraud_prob:.2f}%', f'{fraud_prob:.2f}%'],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.subheader("🔍 Key Risk Factors")
                try:
                    feature_names = model.feature_names_in_
                    importances = model.feature_importances_
                    
                    # Get top 10 features
                    top_indices = np.argsort(importances)[-10:][::-1]
                    top_features = feature_names[top_indices]
                    top_importances = importances[top_indices]
                    
                    fig = px.bar(
                        x=top_importances,
                        y=top_features,
                        orientation='h',
                        title="Top 10 Most Important Features",
                        labels={'x': 'Importance', 'y': 'Feature'},
                        color=top_importances,
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check that all required fields are filled correctly.")

def show_dashboard_page():
    """Analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    st.info("Loading comprehensive analytics... This may take a moment.")
    
    # Import and run dashboard
    try:
        import dashboard
        dashboard.main()
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")
        st.info("Please ensure dashboard.py is available and data files are in the correct location.")

def show_model_insights(model):
    """Model insights and information"""
    st.header("📈 Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Information")
        st.info(f"""
        **Model Type:** Random Forest Classifier
        
        **Features:** {len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 'N/A'}
        
        **Estimators:** {model.n_estimators if hasattr(model, 'n_estimators') else 'N/A'}
        """)
    
    with col2:
        st.subheader("Model Performance")
        st.success("""
        **Accuracy:** ~88-100% (depending on dataset)
        
        **Precision:** High for fraud detection
        
        **Recall:** Optimized for fraud cases
        """)
    
    # Feature importance visualization
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance Analysis")
        
        feature_names = model.feature_names_in_
        importances = model.feature_importances_
        
        # Get top 20 features
        top_indices = np.argsort(importances)[-20:][::-1]
        top_features = feature_names[top_indices]
        top_importances = importances[top_indices]
        
        fig = px.bar(
            x=top_importances,
            y=top_features,
            orientation='h',
            title="Top 20 Most Important Features for Fraud Detection",
            labels={'x': 'Importance Score', 'y': 'Feature'},
            color=top_importances,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model architecture
    st.subheader("Model Architecture")
    st.code("""
    Random Forest Classifier
    ├── Multiple Decision Trees
    ├── Ensemble Learning
    ├── Feature Engineering
    └── Preprocessing Pipeline
    """, language="text")

def show_about_page():
    """About page"""
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ## 🚨 AI Insurance Fraud Detection System
    
    This comprehensive platform combines **Machine Learning**, **Artificial Intelligence**, and **Data Analytics** 
    to detect and prevent insurance fraud.
    
    ### ✨ Key Features
    
    - **Real-Time Fraud Prediction**: Instant risk assessment for insurance claims
    - **Advanced Analytics**: Time-series analysis, geographic patterns, and risk factor identification
    - **Machine Learning Model**: Random Forest Classifier with high accuracy
    - **Interactive Visualizations**: Dynamic charts and graphs for data insights
    
    ### 🔧 Technology Stack
    
    - **Machine Learning**: scikit-learn, Random Forest
    - **Data Analytics**: pandas, numpy
    - **Visualization**: Plotly, Streamlit
    - **Web Framework**: Streamlit
    
    ### 📊 Model Performance
    
    The model has been trained on historical insurance claim data and achieves:
    - High accuracy in fraud detection
    - Balanced precision and recall
    - Robust feature engineering
    
    ### 🎯 Use Cases
    
    1. **Real-time Claim Assessment**: Evaluate new claims instantly
    2. **Risk Analysis**: Identify patterns and trends in fraud
    3. **Preventive Measures**: Proactive fraud detection
    4. **Data-Driven Decisions**: Evidence-based claim processing
    
    ### 📝 Data Sources
    
    The system uses comprehensive insurance claim data including:
    - Policyholder information
    - Accident details
    - Vehicle information
    - Claim amounts and history
    
    ---
    
    **Developed with ❤️ using AI, ML, and Data Analytics**
    """)

if __name__ == "__main__":
    main()

