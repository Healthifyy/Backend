import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Paths relative to project root
DATASET_PATH = "ml/datasets/dataset.csv"
SEVERITY_CSV_PATH = "ml/datasets/symptom_severity.csv"
MODEL_PKL = "ml/healthify_model.pkl"
ENCODER_PKL = "ml/disease_encoder.pkl"
COLUMNS_PKL = "ml/symptom_columns.pkl"
SEVERITY_JSON = "ml/symptom_severity.json"

def main():
    print("--- 1. LOADING DATA ---")
    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Strip whitespace from column names and Disease values
    df.columns = df.columns.str.strip()
    df['Disease'] = df['Disease'].str.strip()
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\n--- 2. PREPARING DATA ---")
    X = df.drop('Disease', axis=1)
    y = df['Disease']
    
    # Fill NaN symptoms with 0 or something similar if columns are sparse
    # The dataset usually has Symptom_1, Symptom_2... until Symptom_17.
    # We need to pivot or handle the categorical symptoms.
    # NOTE: The provided Kaggle dataset format usually requires pivoting or encoding.
    # However, the user request says 'X = all columns except Disease'.
    # If the CSV is in the format Disease, Symptom_1, Symptom_2... we might need to 
    # transform it to a binary matrix where each column is one unique symptom.
    
    # Let's get unique symptoms
    all_symptoms = []
    for col in X.columns:
        all_symptoms.extend(df[col].dropna().unique())
    
    unique_symptoms = sorted(list(set([s.strip().lower() for s in all_symptoms if isinstance(s, str)])))
    
    # Create the training matrix
    X_processed = pd.DataFrame(0, index=np.arange(len(df)), columns=unique_symptoms)
    
    for i in range(len(df)):
        row_symptoms = df.iloc[i].drop('Disease').dropna().values
        for s in row_symptoms:
            s_clean = s.strip().lower().replace(" ", "_") # Match severity formatting
            if s_clean in X_processed.columns:
                X_processed.at[i, s_clean] = 1
            # Fallback for original string
            elif s.strip().lower() in X_processed.columns:
                X_processed.at[i, s.strip().lower()] = 1

    # Update symptom columns list to the processed ones
    symptom_columns = list(X_processed.columns)
    
    # LabelEncoder for y
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save artifacts
    with open(ENCODER_PKL, 'wb') as f:
        pickle.dump(le, f)
    with open(COLUMNS_PKL, 'wb') as f:
        pickle.dump(symptom_columns, f)
        
    print(f"Diseases: {len(le.classes_)} | Symptoms: {len(symptom_columns)}")

    print("\n--- 3. TRAINING MODEL ---")
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\n--- 4. SAVING MODEL ---")
    with open(MODEL_PKL, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PKL}")

    print("\n--- 5. SYMPTOM SEVERITY ---")
    try:
        sev_df = pd.read_csv(SEVERITY_CSV_PATH)
        # Strip whitespace, replace spaces with underscores, lowercase
        sev_df['Symptom'] = sev_df['Symptom'].str.strip().str.lower().str.replace(" ", "_")
        severity_dict = dict(zip(sev_df['Symptom'], sev_df['weight']))
        
        with open(SEVERITY_JSON, 'w') as f:
            json.dump(severity_dict, f, indent=4)
        print("Symptom severity data saved")
    except Exception as e:
        print(f"Error processing severity: {e}")

    print("\n--- 6. FINAL SUMMARY ---")
    print(f"Final Model Accuracy: {accuracy:.4f}")
    print(f"Number of diseases: {len(le.classes_)}")
    print(f"Number of symptom features: {len(symptom_columns)}")
    print(f"Pickle files created: {MODEL_PKL}, {ENCODER_PKL}, {COLUMNS_PKL}")

if __name__ == "__main__":
    main()
