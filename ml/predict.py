import pickle
import numpy as np
import os
import sys

# Global variables for model state
_model = None
_symptom_columns = None
_disease_encoder = None

# Paths to artifacts
MODEL_PKL = "ml/healthify_model.pkl"
COLUMNS_PKL = "ml/symptom_columns.pkl"
ENCODER_PKL = "ml/disease_encoder.pkl"

# EMERGENCY and URGENT symptom sets
EMERGENCY_SYMPTOMS = {
  "chest_pain", "breathlessness", "unconsciousness", "paralysis",
  "altered_sensorium", "weakness_of_one_body_side", "slurred_speech",
  "loss_of_balance", "blood_in_sputum"
}

URGENT_SYMPTOMS = {
  "high_fever", "sudden_high_fever", "fast_heart_rate",
  "neck_stiffness", "stiff_neck", "severe_headache"
}

# DISEASE to TEST mapping
DISEASE_TESTS = {
  "Pneumonia": ["Chest X-Ray", "CBC", "CRP", "Sputum Culture"],
  "Heart attack": ["ECG", "Troponin", "CBC", "Echo"],
  "Bronchial Asthma": ["Spirometry", "Chest X-Ray", "Peak Flow"],
  "Typhoid": ["Widal Test", "Blood Culture", "CBC"],
  "Dengue": ["NS1 Antigen", "Dengue IgM/IgG", "CBC", "Platelet Count"],
  "Malaria": ["Peripheral Blood Smear", "RDT", "CBC"],
  "Tuberculosis": ["Sputum AFB", "Chest X-Ray", "Mantoux Test"],
  "Diabetes": ["Fasting Blood Glucose", "HbA1c", "OGTT"],
  "Hypertension": ["BP monitoring", "ECG", "Renal function"],
  "Jaundice": ["LFT", "Bilirubin", "Hepatitis Panel"],
  "Urinary tract infection": ["Urine Culture", "Urine R/M", "CBC"],
  "Common Cold": ["Clinical diagnosis only"],
  "Migraine": ["Clinical diagnosis", "CT Scan if first episode"],
  "Gastroenteritis": ["Stool Culture", "CBC", "Electrolytes"],
  "Chicken pox": ["Clinical diagnosis", "Tzanck Smear if needed"]
}

def load_model() -> bool:
    """Load all 3 pickle files into global variables."""
    global _model, _symptom_columns, _disease_encoder
    try:
        if not os.path.exists(MODEL_PKL) or not os.path.exists(COLUMNS_PKL) or not os.path.exists(ENCODER_PKL):
            print(f"Error: One or more model artifacts missing in ml/ folder.")
            return False
            
        with open(MODEL_PKL, "rb") as f:
            _model = pickle.load(f)
        with open(COLUMNS_PKL, "rb") as f:
            _symptom_columns = pickle.load(f)
        with open(ENCODER_PKL, "rb") as f:
            _disease_encoder = pickle.load(f)
            
        print(f"Model loaded: {len(_disease_encoder.classes_)} diseases, {len(_symptom_columns)} symptoms")
        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False

def symptoms_to_vector(symptoms: list) -> np.ndarray:
    """Convert input symptoms into a binary feature vector."""
    vector = np.zeros(len(_symptom_columns))
    for s in symptoms:
        # Normalize: lowercase, strip, replace spaces with underscores
        clean_s = s.lower().strip().replace(" ", "_")
        if clean_s in _symptom_columns:
            idx = _symptom_columns.index(clean_s)
            vector[idx] = 1
    return vector.reshape(1, -1)

def classify_urgency(symptoms: list, existing_conditions: list, 
                     is_pregnant: bool, severity: int) -> tuple[str, str]:
    """Determine case urgency based on clinical rules."""
    # Normalize input symptoms for checking
    clean_symptoms = [s.lower().strip().replace(" ", "_") for s in symptoms]
    
    # 1. Emergency Symptoms check
    for s in clean_symptoms:
        if s in EMERGENCY_SYMPTOMS:
            return ("emergency", f"Emergency symptom detected: {s.replace('_', ' ')}")
            
    # 2. Diabetic high-risk check
    if "diabetes" in [c.lower() for c in existing_conditions]:
        high_risk_s = {"chest_pain", "breathlessness", "high_fever"}
        for s in clean_symptoms:
            if s in high_risk_s:
                return ("emergency", "Diabetic patient with high-risk symptoms")
                
    # 3. Pregnancy fever check
    if is_pregnant and "high_fever" in clean_symptoms:
        return ("emergency", "Pregnant patient with fever requires immediate attention")
        
    # 4. Urgent symptoms check
    for s in clean_symptoms:
        if s in URGENT_SYMPTOMS:
            return ("urgent", f"Urgent symptom detected: {s.replace('_', ' ')}")
            
    # 5. High severity check
    if severity >= 8:
        return ("urgent", "High severity score reported")
        
    # 6. Default routine
    return ("routine", "No emergency indicators detected")

def get_red_flags(symptoms: list, existing_conditions: list) -> list[str]:
    """Check for dangerous symptom/condition combinations."""
    clean_s = [s.lower().strip().replace(" ", "_") for s in symptoms]
    conds = [c.lower() for c in existing_conditions]
    flags = []
    
    # Combination checks
    if "chest_pain" in clean_s and "breathlessness" in clean_s:
        flags.append("Chest pain with breathlessness — possible cardiac emergency")
    if "high_fever" in clean_s and "neck_stiffness" in clean_s:
        flags.append("High fever with neck stiffness — possible meningitis")
    if "high_fever" in clean_s and "altered_sensorium" in clean_s:
        flags.append("Fever with confusion — possible sepsis or meningitis")
    if "diabetes" in conds and "high_fever" in clean_s:
        flags.append("Diabetic patient with fever — elevated infection complication risk")
    if "diabetes" in conds and "chest_pain" in clean_s:
        flags.append("Diabetic patient with chest pain — silent cardiac event possible")
        
    # Single symptom checks
    if "blood_in_sputum" in clean_s:
        flags.append("Blood in sputum — requires immediate investigation")
    if "unconsciousness" in clean_s:
        flags.append("Loss of consciousness — call emergency services immediately")
        
    return flags

def predict_disease(symptoms: list, existing_conditions: list = [], 
                    is_pregnant: bool = False, severity: int = 5) -> dict:
    """Main entry point for AI disease prediction and triage."""
    # 1. Ensure model is loaded
    if _model is None:
        if not load_model():
            return {"error": "Model files missing or corrupt"}
            
    # 2. Convert to vector
    vector = symptoms_to_vector(symptoms)
    
    # 3. & 4. Run prediction and top 3
    probs = _model.predict_proba(vector)[0]
    top_indices = np.argsort(probs)[::-1][:3]
    
    # 5, 6, 7. Process top diseases
    top_conditions = []
    for idx in top_indices:
        disease_name = _disease_encoder.classes_[idx]
        score = int(probs[idx] * 100)
        
        # Confidence logic
        if score >= 60: confidence = "high"
        elif score >= 35: confidence = "medium"
        else: confidence = "low"
        
        top_conditions.append({
            "name": disease_name,
            "confidence": confidence,
            "match_score": score,
            "reasoning": "Matched based on symptom pattern analysis"
        })
        
    # 8. Triage
    urgency, urgency_reason = classify_urgency(symptoms, existing_conditions, is_pregnant, severity)
    
    # 9. Red flags
    red_flags = get_red_flags(symptoms, existing_conditions)
    
    # 10. Recommended tests
    top_condition_name = top_conditions[0]["name"]
    recommended_tests = DISEASE_TESTS.get(top_condition_name, ["Consult a doctor for specific tests"])
    
    # Formatting doctor summary
    symptom_str = ", ".join(symptoms)
    doctor_summary = (f"Patient presents with {symptom_str}. "
                      f"AI triage identifies {top_condition_name} as most probable. "
                      f"Urgency level: {urgency}. "
                      f"This summary was generated by Healthify AI and is "
                      f"not a replacement for clinical diagnosis.")
    
    return {
        "urgency": urgency,
        "urgency_reason": urgency_reason,
        "top_conditions": top_conditions,
        "red_flags": red_flags,
        "recommended_tests": recommended_tests,
        "home_care": ["Rest and stay hydrated", "Monitor symptoms closely", "Avoid self-medication"],
        "when_to_escalate": [
            "If symptoms worsen significantly", 
            "If new symptoms appear",
            "If fever crosses 103°F / 39.4°C"
        ],
        "doctor_summary": doctor_summary,
        "source": "ml_only"
    }
