import pickle
import numpy as np
import os
import sys

SYMPTOM_ALIASES = {
    "body_ache": "muscle_pain",
    "body ache": "muscle_pain",
    "bodyache": "muscle_pain",
    "sneezing": "continuous_sneezing",
    "fever": "high_fever",
    "stomach_ache": "stomach_pain",
    "eye_redness": "redness_of_eyes",
    "loose_motion": "diarrhoea",
    "loose_stools": "diarrhoea",
    "cold": "runny_nose",
    "sore_throat": "throat_irritation",
    "back ache": "back_pain",
    "backache": "back_pain",
}

def normalize_symptom(s):
    s = s.lower().strip().replace(" ", "_")
    return SYMPTOM_ALIASES.get(s, s)

# Global variables for model state
_model = None
_symptom_columns = None
_disease_encoder = None

# Paths to artifacts
MODEL_PKL = "ml/healthify_model.pkl"
COLUMNS_PKL = "ml/symptom_columns.pkl"
ENCODER_PKL = "ml/disease_encoder.pkl"

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
  "Chicken pox": ["Clinical diagnosis", "Tzanck Smear if needed"],
  "Allergy": ["Allergy testing", "IgE levels"]
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
        clean_s = normalize_symptom(s)
        
        # 3. Exact match
        if clean_s in _symptom_columns:
            idx = _symptom_columns.index(clean_s)
            vector[idx] = 1
            continue
            
        # 4. Partial / fuzzy match
        for col_idx, col_name in enumerate(_symptom_columns):
            if clean_s in col_name or col_name in clean_s:
                vector[col_idx] = 1
                break
                
    return vector.reshape(1, -1)

def classify_urgency(clean_symptoms: list, severity: int, is_pregnant: bool, recent_travel: bool) -> str:
    """Determine case urgency based on clinical rules. OVERRIDES ML"""
    # Emergency rules
    emergency_keywords = {"chest_pain", "breathlessness", "unconsciousness", "paralysis", "slurred_speech", "severe_bleeding"}
    has_emergency = any(k in clean_symptoms for k in emergency_keywords)
    
    if severity >= 7 or has_emergency or ("high_fever" in clean_symptoms and "breathlessness" in clean_symptoms):
        return "EMERGENCY"
        
    # Urgent rules
    urgent_keywords = {"high_fever", "vomiting", "dehydration", "jaundice"}
    has_urgent = any(k in clean_symptoms for k in urgent_keywords)
    has_any_fever = any("fever" in s for s in clean_symptoms)
    
    if severity >= 4 or has_urgent or (is_pregnant and has_any_fever) or recent_travel:
        return "URGENT"
        
    # Routine rules
    return "ROUTINE"

def get_red_flags(clean_s: list, existing_conditions: list) -> list[str]:
    """Check for dangerous symptom/condition combinations."""
    conds = [c.lower() for c in existing_conditions]
    flags = []
    
    if "chest_pain" in clean_s and "breathlessness" in clean_s:
        flags.append("Chest pain with breathlessness — possible cardiac emergency")
    if "high_fever" in clean_s and "neck_stiffness" in clean_s:
        flags.append("High fever with neck stiffness — possible meningitis")
    if "diabetes" in conds and "high_fever" in clean_s:
        flags.append("Diabetic patient with fever — elevated infection complication risk")
        
    return flags

def predict_disease(symptoms: list, existing_conditions: list = [], 
                    is_pregnant: bool = False, severity: int = 5,
                    recent_travel: bool = False, age: int = 30, gender: str = "other") -> dict:
    """Main entry point for AI disease prediction and triage."""
    if _model is None:
        if not load_model():
            return {"error": "Model files missing or corrupt"}
            
    vector = symptoms_to_vector(symptoms)
    
    # Calculate matched symptoms for score and red flags
    matched_symptoms = []
    unmatched_symptoms = []
    for s in symptoms:
        clean_s = normalize_symptom(s)
        if clean_s in _symptom_columns:
            matched_symptoms.append(clean_s)
        else:
            found = False
            for col_name in _symptom_columns:
                if clean_s in col_name or col_name in clean_s:
                    matched_symptoms.append(col_name)
                    found = True
                    break
            if not found:
                unmatched_symptoms.append(s)
                
    total_input = len(symptoms)
    match_score = int((len(matched_symptoms) / total_input) * 100) if total_input > 0 else 0
    
    probs = _model.predict_proba(vector)[0]
    
    # PRIORITY BOOST RULES
    for i, disease in enumerate(_disease_encoder.classes_):
        boost = 0.0
        
        has_chest_pain = "chest_pain" in matched_symptoms
        has_breath = "breathlessness" in matched_symptoms
        has_fever = "high_fever" in matched_symptoms
        has_headache = "headache" in matched_symptoms
        has_body_ache = "body_ache" in matched_symptoms or "muscle_pain" in matched_symptoms
        has_runny = "runny_nose" in matched_symptoms
        has_sneezing = "sneezing" in matched_symptoms or "continuous_sneezing" in matched_symptoms
        has_yellow = "yellowish_skin" in matched_symptoms
        has_dark_urine = "dark_urine" in matched_symptoms
        has_vomiting = "vomiting" in matched_symptoms
        has_diarrhoea = "diarrhoea" in matched_symptoms
        has_dehydration = "dehydration" in matched_symptoms
        
        if has_chest_pain and has_breath:
            if disease == "Heart attack": boost += 0.3
        if has_fever and has_headache and has_body_ache:
            if disease == "Malaria": boost += 0.4
            elif disease == "Dengue": boost += 0.35
            elif disease == "Typhoid": boost += 0.3
        if has_runny and has_sneezing:
            if disease in ["Common Cold", "Allergy"]: boost += 0.3
        if has_yellow and has_dark_urine:
            if "hepatitis" in disease.lower() or disease == "Jaundice": boost += 0.3
        if has_vomiting and has_diarrhoea and has_dehydration:
            if disease in ["Cholera", "Gastroenteritis"]: boost += 0.3
            
        if is_pregnant and has_fever:
            if disease == "Malaria": boost += 0.2
            
        probs[i] = min(probs[i] + boost, 0.99)
    
    top_indices = np.argsort(probs)[::-1][:3]
    
    top_conditions = []
    for idx in top_indices:
        disease_name = _disease_encoder.classes_[idx]
        prob = probs[idx]
        
        if prob >= 0.50: 
            confidence = "High"
        elif prob >= 0.25: 
            confidence = "Medium"
        else: 
            if match_score >= 80:
                confidence = "Medium"
            else:
                confidence = "Low"
        
        top_conditions.append({
            "name": disease_name,
            "confidence": confidence,
            "match_score": match_score,
            "reasoning": f"Probability: {prob:.2f}"
        })
        
    # Urgency Override
    urgency = classify_urgency(matched_symptoms, severity, is_pregnant, recent_travel)
    urgency_reason = "Determined by clinical rule engine"
    
    red_flags = get_red_flags(matched_symptoms, existing_conditions)
    top_condition_name = top_conditions[0]["name"]
    recommended_tests = DISEASE_TESTS.get(top_condition_name, ["Consult a doctor for specific tests"])
    
    doctor_summary = (f"Patient presents with {', '.join(symptoms)}. "
                      f"AI triage identifies {top_condition_name} as most probable. "
                      f"Urgency level: {urgency}.")
    
    return {
        "urgency": urgency,
        "urgency_reason": urgency_reason,
        "top_conditions": top_conditions,
        "red_flags": red_flags,
        "recommended_tests": recommended_tests,
        "home_care": ["Rest and stay hydrated"],
        "when_to_escalate": ["If symptoms worsen significantly", "If new symptoms appear"],
        "doctor_summary": doctor_summary,
        "source": "ml_only"
    }
