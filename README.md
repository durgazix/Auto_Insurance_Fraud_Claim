# Auto Insurance Fraud Claim Detection

## 1. Project idea

This project builds an end-to-end machine learning solution for detecting potentially fraudulent auto insurance claims. The business goal is to help insurers review suspicious claims faster, reduce financial losses, and improve claim assessment efficiency.

The solution combines:
- a trained fraud detection model,
- a Streamlit web application for real-time claim scoring, and
- an analytics dashboard for exploring fraud trends and risk patterns.

## 2. End-to-end project flow

The project follows a complete ML workflow from raw data to a usable application.

1. Data collection
   - The project uses multiple CSV files in the [data](data) folder.
   - The datasets contain insurance policy, accident, vehicle, and claim information.
   - The data dictionary is available in [Data Dictionary for Auto Insurance.txt](Data%20Dictionary%20for%20Auto%20Insurance.txt).

2. Data understanding and exploration
   - The notebook [Auto_Insurance_Fraud_Claims.ipynb](Auto_Insurance_Fraud_Claims.ipynb) was used to inspect the dataset and understand fraud-related patterns.
   - Key fields include age, policy details, vehicle cost, accident severity, annual mileage, claim amount, and fraud labels.

3. Data preprocessing
   - The preprocessing pipeline in [utils/preprocess.py](utils/preprocess.py) prepares raw input data for the trained model.
   - Steps include:
     - removing non-feature columns such as claim ID, policy number, vehicle registration, and checkpoint fields,
     - dropping date columns,
     - converting categorical values using one-hot encoding,
     - aligning the transformed data to the exact feature columns expected by the model.

4. Model training
   - A Random Forest Classifier was trained in the notebook using a stratified train-test split.
   - The model was saved as [model/best_fraud_model.pkl](model/best_fraud_model.pkl).
   - Feature names used during inference are stored in [model/feature_columns.pkl](model/feature_columns.pkl).

5. Model evaluation
   - The evaluation report in [File01_Evaluation_Report.txt](File01_Evaluation_Report.txt) shows:
     - Accuracy: 0.8842
     - ROC AUC: 0.9600
   - The model performs well overall, especially for identifying non-fraud cases, while still flagging a meaningful subset of suspicious claims.

6. Deployment and prediction interface
   - The app [app_comprehensive.py](app_comprehensive.py) loads the trained model and allows users to enter claim details and receive a fraud risk score.
   - The dashboard [dashboard.py](dashboard.py) provides analytics such as fraud trends over time, geographic patterns, and risk-factor analysis.

## 3. What has already been done

The repository already contains the main building blocks of a complete fraud detection system:

- Data files in [data](data)
- Data dictionary and evaluation report
- Training notebook for model development
- Preprocessing utilities
- Trained model artifacts in [model](model)
- Streamlit-based prediction app
- Analytics dashboard
- Dependency list in [requirements.txt](requirements.txt)

## 4. Project structure

```text
.
├── app_comprehensive.py        # Main Streamlit fraud prediction app
├── dashboard.py                # Analytics dashboard
├── Auto_Insurance_Fraud_Claims.ipynb  # Model training notebook
├── Data Dictionary for Auto Insurance.txt
├── File01_Evaluation_Report.txt
├── MODEL_EXPLANATION.md        # Notes about the trained model artifact
├── requirements.txt            # Python dependencies
├── data/
│   ├── Auto_Insurance_Fraud_Claims_File01.csv
│   ├── Auto_Insurance_Fraud_Claims_File02.csv
│   └── Auto_Insurance_Fraud_Claims_File002.csv
├── model/
│   ├── best_fraud_model.pkl    # Trained fraud detection model
│   └── feature_columns.pkl     # Expected feature order for inference
└── utils/
    └── preprocess.py            # Preprocessing logic
```

## 5. Model creation process

The model creation process in this project is structured as follows:

1. Prepare the dataset
   - Load the insurance claim CSV files.
   - Combine them into a single training-ready dataset.
   - Identify the target column, which indicates whether a claim is fraud (Y/N).

2. Build features
   - Use policy, accident, vehicle, demographic, and claim-related fields.
   - Convert categorical fields into numeric form through one-hot encoding.
   - Remove columns that are identifiers or would leak information during inference.

3. Split the data
   - The notebook uses a stratified train-test split so that the train and test sets preserve the fraud ratio.

4. Train the model
   - A Random Forest Classifier is used.
   - The model is trained with balanced class weighting to handle the imbalance between fraud and non-fraud claims.

5. Save the trained model
   - The trained model is serialized using joblib and stored in [model/best_fraud_model.pkl](model/best_fraud_model.pkl).
   - The final feature structure is saved in [model/feature_columns.pkl](model/feature_columns.pkl).

## 6. How the model works

The fraud model is a Random Forest Classifier.

### What it does
- It reads a new claim record.
- It transforms the record into the same feature structure used during training.
- It predicts whether the claim is likely fraudulent or legitimate.
- It also returns a probability score that shows how confident the model is.

### Why Random Forest was chosen
- It handles tabular data well.
- It captures non-linear relationships between features.
- It reduces overfitting compared with a single decision tree.
- It is robust for classification tasks like fraud detection.

### How predictions are made
- The model builds many decision trees.
- Each tree votes for a class (fraud or not fraud).
- The forest combines those votes and produces a final prediction and probability.

### Important note
- The model depends on the preprocessing pipeline and the saved feature list. If the input format changes, the model may fail or give incorrect results.

## 7. How the application works

### Fraud prediction app
The main app, [app_comprehensive.py](app_comprehensive.py), provides a user-friendly interface where a user can enter claim details such as:
- gender,
- age,
- occupation,
- accident severity,
- vehicle cost,
- annual mileage,
- total claim amount.

The app then:
1. builds a small dataframe from the entered values,
2. preprocesses the data,
3. loads the trained model,
4. predicts fraud probability,
5. displays the result as a fraud/no-fraud assessment with a risk score.

### Analytics dashboard
The dashboard, [dashboard.py](dashboard.py), helps analysts understand patterns in the data by showing:
- monthly fraud trends,
- states with higher fraud concentration,
- fraud rates by age group,
- fraud rates by accident severity,
- fraud rates by vehicle cost or claim amount.

## 8. How to start this project locally

### Prerequisites
- Python 3.9+ recommended
- pip

### Step 1: Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the prediction app

```bash
streamlit run app_comprehensive.py
```

### Step 4: Run the analytics dashboard

```bash
streamlit run dashboard.py
```

## 9. Suggested next steps and future roadmap

To take this project from a strong prototype to a high-quality, production-ready fraud detection platform, the following major improvements are planned and should be tracked over time.

### 9.1 Improve model performance
- Retrain the model with more advanced feature engineering.
- Compare multiple algorithms such as XGBoost, LightGBM, and CatBoost.
- Optimize hyperparameters using cross-validation.
- Evaluate using precision, recall, F1-score, ROC AUC, and PR AUC instead of relying only on accuracy.
- Address class imbalance more effectively to improve fraud detection sensitivity.

### 9.2 Strengthen the preprocessing pipeline
- Handle missing values more robustly.
- Detect and manage outliers.
- Improve categorical handling and consistency.
- Create new business-relevant features such as:
  - claim-to-vehicle-cost ratio,
  - policy duration,
  - mileage deviation,
  - claim frequency patterns,
  - accident severity weighting.

### 9.3 Add explainability and trust
- Add SHAP-based explanations for each prediction.
- Show the top reasons why the model flagged a claim as suspicious.
- Make the system more transparent for investigators and business users.

### 9.4 Make the application more professional
- Improve input validation in the Streamlit app.
- Add better error handling and user guidance.
- Display confidence scores and explanation panels clearly.
- Add a prediction history view so users can review prior claims.
- Improve the visual design of the dashboard and app layout.

### 9.5 Add testing and reliability
- Create automated tests for preprocessing logic.
- Test that the model receives the correct feature structure.
- Add unit tests for prediction behavior and data transformations.
- Ensure the project is reproducible and stable across environments.

### 9.6 Prepare for deployment
- Convert the current app into a production-ready backend API using FastAPI or Flask.
- Containerize the application with Docker.
- Add logging, monitoring, and error tracking.
- Version the model and preprocessing pipeline for better maintainability.

### 9.7 Add MLOps and versioning
- Track model version, feature version, and dataset version.
- Store predictions and feedback for retraining.
- Create a workflow for updating the model safely.
- Add experiment tracking so different model versions can be compared.

### 9.8 Add human-in-the-loop review
- Build a review workflow where analysts can approve, reject, or investigate flagged claims.
- Add manual override options.
- Create a queue for uncertain predictions so human reviewers can make the final judgment.

### 9.9 Improve business impact and usability
- Turn the project into a practical decision-support tool for claim investigators.
- Show how the model helps reduce false positives and prioritizes suspicious claims.
- Connect the solution to real insurance workflows rather than treating it as a standalone demo.

### 9.10 Final goal: reach a 10/10 project standard
The long-term goal is to transform this repository into a polished, production-style fraud detection platform that is:
- highly accurate,
- explainable,
- tested,
- deployable,
- and useful in a real insurance environment.

## 10. Summary

This project is a complete fraud detection solution that starts with raw insurance claim data and ends with an interactive application that can assist insurers in flagging suspicious claims. It already includes data understanding, preprocessing, model training, model evaluation, and deployment into a Streamlit-based user interface.

In short, the project demonstrates how machine learning can be used to turn historical claim data into a practical fraud-risk decision support system.
