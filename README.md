# 🚗 Auto Insurance Fraud Detection - ML Web App

This project is an end-to-end AI solution to detect fraudulent auto insurance claims using machine learning. It includes:

- 🧠 ML models (Random Forest, XGBoost, etc.)
- ⚙️ Flask API to serve the trained model
- 🎨 Streamlit frontend for interactive fraud prediction
- 📊 Data preprocessing, feature engineering & evaluation
- 📁 Trained model & column structure for consistent inference

---

## 📁 Folder Structure

project/
├── app.py # Flask backend API
├── streamlit_app.py # Streamlit frontend UI
├── model/
│ ├── best_fraud_model.pkl # Trained ML model
│ └── feature_columns.pkl # List of one-hot encoded features
├── utils/
│ └── preprocess.py # Reusable preprocessing logic
├── requirements.txt # Dependencies
├── README.md # Project overview
└── data/ # (Optional) CSV files


---

## 📦 Features

- Predict if a given auto insurance claim is **Fraudulent or Not**
- Preprocessing with `LabelEncoding`, `One-Hot Encoding`, etc.
- Supports unseen test data files: `File01`, `File03`, `Results_Submission.csv`
- Evaluation using **Accuracy**, **Recall**, **AUC**, **Confusion Matrix**
- Extendable with 📍 fraud heatmaps, clustering (KMeans), and RAG

---

## 🧪 ML Models Used

| Model                | Metrics Tracked       |
|----------------------|-----------------------|
| ✅ Random Forest     | F1, Recall, AUC       |
| ✅ XGBoost           | F1, Recall, AUC       |
| ✅ Logistic Regression | Baseline Model     |
| ✅ Decision Tree     | Easy Interpretable    |
| ✅ MLP Classifier    | Advanced (Neural Net) |

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/auto-insurance-fraud.git
cd auto-insurance-fraud

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

Install Requirements
pip install -r requirements.txt

Run Flask Backend
python app.py

Run Streamlit Frontend
streamlit run streamlit_app.py

Example API Payload
POST /predict
Content-Type: application/json

{
  "Age_Insured": 45,
  "Gender": "male",
  "Occupation": "engineer",
  "Accident_Type": "Rear Collision",
  ...
}


---

Would you like this turned into a downloadable file? Or do you want it tailored for **HuggingFace Spaces**, **Render**, or **GitHub Pages** deployment as well?
