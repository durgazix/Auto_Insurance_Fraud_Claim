<<<<<<< HEAD
# 🚨 AI Insurance Fraud Detection System

A comprehensive Machine Learning and Data Analytics platform for detecting insurance fraud using advanced AI techniques.

## ✨ Features

### 🔮 Real-Time Fraud Prediction
- Instant fraud risk assessment for insurance claims
- Probability-based predictions with detailed risk scoring
- Interactive input forms with validation
- Visual probability displays

### 📊 Advanced Analytics Dashboard
- **Time-Series Analysis**: Monthly fraud trends and patterns
- **Geographic Analysis**: State-wise fraud distribution
- **Risk Factor Analysis**: Age, severity, vehicle cost, and claim amount analysis
- **Interactive Visualizations**: Dynamic charts using Plotly
- **Key Insights**: Automated pattern detection and insights

### 📈 Model Insights
- Feature importance visualization
- Model performance metrics
- Architecture details

## 🛠️ Technology Stack

- **Machine Learning**: scikit-learn (Random Forest Classifier)
- **Data Analytics**: pandas, numpy
- **Visualization**: Plotly, Streamlit
- **Web Framework**: Streamlit

## 📁 Project Structure

```
Learnathon/
├── app_comprehensive.py        # Main Streamlit application
├── dashboard.py                # Analytics dashboard module
├── requirements.txt            # Python dependencies
├── model/
│   ├── best_fraud_model.pkl   # Trained ML model
│   └── feature_columns.pkl    # Feature column mapping
├── utils/
│   └── preprocess.py          # Data preprocessing utilities
└── data/                      # Data files directory
    ├── Auto_Insurance_Fraud_Claims_File01.csv
    ├── Auto_Insurance_Fraud_Claims_File002.csv
    └── ...
```

## 🚀 Getting Started

### Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are in place**:
   - `model/best_fraud_model.pkl`
   - `model/feature_columns.pkl`

### Running the Application

#### Option 1: Comprehensive Streamlit App (Recommended)
```bash
streamlit run app_comprehensive.py
```
This launches the full-featured application with:
- Fraud prediction interface
- Analytics dashboard
- Model insights
- About page

#### Option 2: Analytics Dashboard Only
```bash
streamlit run dashboard.py
```
This launches only the analytics dashboard.

## 📊 Model Information

### Model Type
- **Algorithm**: Random Forest Classifier
- **Features**: 644 engineered features
- **Performance**: 
  - Accuracy: ~88-100% (depending on dataset)
  - High precision and recall for fraud detection

### Model Training
The model was trained on historical insurance claim data with:
- Feature engineering
- Data preprocessing
- Balanced class weights
- Cross-validation

## 🎯 Usage Examples

### Making a Prediction

1. Launch the Streamlit app
2. Navigate to "🔮 Fraud Prediction"
3. Fill in the claim details:
   - Personal information (age, gender, education, occupation)
   - Accident details (severity, type)
   - Vehicle information (cost, mileage)
   - Claim amount
4. Click "🔍 Predict Fraud Risk"
5. View the prediction with probability scores and risk assessment

### Viewing Analytics

1. Navigate to "📊 Analytics Dashboard"
2. Use filters to analyze specific time periods or states
3. Explore:
   - Time-series trends
   - Geographic patterns
   - Risk factor correlations
   - Key insights

## 📈 Key Metrics Tracked

- Total claims count
- Fraud cases and fraud rate
- Average claim amounts
- Monthly fraud trends
- State-wise fraud distribution
- Risk factor analysis (age, severity, cost, claim amount)

## 🔍 Data Sources

The system uses comprehensive insurance claim data including:
- Policyholder demographics
- Accident details and severity
- Vehicle information
- Claim amounts and history
- Geographic data

## 🧪 Model Verification

The model has been verified and tested:
- ✅ Preprocessing pipeline works correctly
- ✅ Model loads and predicts successfully
- ✅ Feature alignment verified
- ✅ Performance metrics validated

## 📝 Notes

- The model uses one-hot encoding for categorical features
- Missing features are automatically filled with zeros
- Date columns are excluded from model input
- The preprocessing pipeline ensures feature consistency

## 🤝 Contributing

This is a learning project demonstrating:
- Machine Learning model deployment
- Data Analytics and visualization
- Interactive web applications
- Fraud detection systems

## 📄 License

This project is for educational and demonstration purposes.

---

**Developed with ❤️ using AI, ML, and Data Analytics**