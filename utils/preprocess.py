import pandas as pd
import joblib

def preprocess(df):
    drop_cols = ['Claim_ID', 'Policy_Num', 'Vehicle_Registration', 'Check_Point']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    datetime_cols = [col for col in df.columns if 'Date' in col]
    df = df.drop(columns=datetime_cols, errors='ignore')

    df = pd.get_dummies(df)

    # Load the correct feature structure
    expected_cols = joblib.load("model/feature_columns.pkl")

    # Add missing columns with 0 (optimized)
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        missing_df = pd.DataFrame(0, index=df.index, columns=list(missing_cols))
        df = pd.concat([df, missing_df], axis=1)

    # Drop extra columns and reorder
    df = df[expected_cols]
    
    return df
