# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# from utils.preprocess import preprocess 
# app = Flask(__name__)


# def preprocess(df):
#     drop_cols = ['Claim_ID', 'Policy_Num', 'Vehicle_Registration', 'Check_Point']
#     df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

#     datetime_cols = [col for col in df.columns if 'Date' in col]
#     df = df.drop(columns=datetime_cols, errors='ignore')

#     # Encode categorical
#     df = pd.get_dummies(df)

#     return df
# model = joblib.load("model/best_fraud_model.pkl")

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     df = pd.DataFrame([data])
#     df_clean = preprocess(df)
    
#     prediction = model.predict(df_clean)[0]
#     label = "Fraud" if prediction == 1 else "Not Fraud"
    
#     return jsonify({"prediction": label})

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, jsonify
import joblib
import pandas as pd
from utils.preprocess import preprocess

app = Flask(__name__)

model = joblib.load("model/best_fraud_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    df_clean = preprocess(df)

    prediction = model.predict(df_clean)[0]
    label = "Fraud" if prediction == 1 else "Not Fraud"

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
