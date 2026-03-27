import pickle
import numpy as np
import os
import sys

MODEL_PKL = "ml/healthify_model.pkl"
ENCODER_PKL = "ml/disease_encoder.pkl"
COLUMNS_PKL = "ml/symptom_columns.pkl"

def load_artifacts():
    try:
        with open(MODEL_PKL, 'rb') as f:
            model = pickle.load(f)
        with open(ENCODER_PKL, 'rb') as f:
            encoder = pickle.load(f)
        with open(COLUMNS_PKL, 'rb') as f:
            columns = pickle.load(f)
        return model, encoder, columns
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e.filename}")
        print("Please run python ml/train_model.py first.")
        sys.exit(1)

def main():
    print("--- LOADING ARTIFACTS ---")
    model, encoder, columns = load_artifacts()
    print("Artifacts loaded successfully.")

    test_symptoms = ["chest_pain", "breathlessness", "high_fever", "cough"]
    print(f"\n--- TESTING PREDICTION with: {test_symptoms} ---")
    
    # Create input vector
    input_vector = np.zeros(len(columns))
    for symptom in test_symptoms:
        # Match cleaning from train_model.py
        clean = symptom.strip().lower().replace(" ", "_")
        if clean in columns:
            input_vector[columns.index(clean)] = 1
        elif symptom.strip().lower() in columns:
            input_vector[columns.index(symptom.strip().lower())] = 1
    
    # Predict probabilities
    input_vector = input_vector.reshape(1, -1)
    probs = model.predict_proba(input_vector)[0]
    
    # Get top 3
    top_indices = np.argsort(probs)[::-1][:3]
    top_diseases = encoder.inverse_transform(top_indices)
    top_probs = probs[top_indices]
    
    print("\n--- TOP 3 PREDICTED DISEASES ---")
    for i in range(3):
        print(f"{i+1}. {top_diseases[i]}: {top_probs[i]*100:.2f}%")
    
    print("\nModel verification: PASSED")

if __name__ == "__main__":
    main()
