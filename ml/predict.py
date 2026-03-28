import pickle
import numpy as np
import json
import os
import sys
import pandas as pd

from ml.medicine_data import get_medicine_info
from ml.knowledge_engine import (
    calculate_confidence,
    get_urgency,
    filter_age_inappropriate_diseases,
    build_reasoning,
    get_age_group,
    get_duration_category,
    has_minimum_symptom_match,
    DISEASE_COMMONALITY_RANK
)

SYMPTOM_LIST = []

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
_disease_symptoms = {}

# Paths to artifacts
MODEL_PKL = "ml/healthify_model.pkl"
COLUMNS_PKL = "ml/symptom_columns.pkl"
ENCODER_PKL = "ml/disease_encoder.pkl"
DS_JSON = "ml/disease_symptoms.json"

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
    global _model, _symptom_columns, _disease_encoder, _disease_symptoms
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
            
        if os.path.exists(DS_JSON):
            with open(DS_JSON, "r") as f:
                _disease_symptoms = json.load(f)
                
        # Populate exported SYMPTOM_LIST for external use
        global SYMPTOM_LIST
        SYMPTOM_LIST.clear()
        SYMPTOM_LIST.extend(list(_symptom_columns))
                
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

def validate_symptoms(symptoms: list) -> list:
    """Filter symptoms against the model's known symptom list."""
    if _symptom_columns is None:
        load_model()
        
    valid = []
    for s in symptoms:
        clean_s = normalize_symptom(s)
        if clean_s in _symptom_columns:
            valid.append(clean_s)
            continue
            
        # Also check for partial matches in the known columns
        for col_name in _symptom_columns:
            if clean_s in col_name or col_name in clean_s:
                valid.append(col_name)
                break
    return list(set(valid)) # Return unique matches

def get_symptom_list():
    """Return the list of columns the model was trained on."""
    return _symptom_columns or []

def classify_urgency(clean_symptoms: list, severity: int, is_pregnant: bool, recent_travel: bool) -> str:
    """Determine case urgency based on clinical rules. OVERRIDES ML"""
    has_fever = any("fever" in s for s in clean_symptoms)
    has_high_fever = "high_fever" in clean_symptoms
    chest_and_breath = "chest_pain" in clean_symptoms and "breathlessness" in clean_symptoms
    emergency_keywords = {"unconsciousness", "paralysis", "slurred_speech", "severe_bleeding"}
    has_emergency_keyword = any(k in clean_symptoms for k in emergency_keywords)
    
    # Emergency rules
    if severity >= 8 or chest_and_breath or has_emergency_keyword:
        return "EMERGENCY"
        
    # Urgent rules
    other_than_high_fever = sum(1 for s in clean_symptoms if s != "high_fever")
    is_preg_and_fever = is_pregnant and has_fever
    breath_alone = "breathlessness" in clean_symptoms
    vomit_and_diarrhoea = "vomiting" in clean_symptoms and "diarrhoea" in clean_symptoms
    
    if severity >= 7 or (has_high_fever and other_than_high_fever >= 2) or is_preg_and_fever or breath_alone or vomit_and_diarrhoea:
        return "URGENT"
        
    # Routine rules
    high_fever_alone = has_high_fever and len(clean_symptoms) == 1
    vomiting_alone = "vomiting" in clean_symptoms and len(clean_symptoms) == 1
    headache_and_fever = "headache" in clean_symptoms and has_fever
    
    # explicit NON_URGENT override for mild cases
    mild_keywords = {"itching", "runny_nose", "sneezing", "mild_fever", "rash", "skin_rash"}
    is_mild_alone = len(clean_symptoms) == 1 and any(k in clean_symptoms[0] for k in mild_keywords)
    skin_rashes_no_fever = any("rash" in s for s in clean_symptoms) and not has_fever
    single_mild_sym = len(clean_symptoms) == 1 and severity < 6
    
    if (severity <= 3) or is_mild_alone or skin_rashes_no_fever or single_mild_sym:
        return "NON_URGENT"
        
    if (4 <= severity <= 6) or high_fever_alone or vomiting_alone or headache_and_fever:
        return "ROUTINE"
        
    # Non-Urgent rules
    return "NON_URGENT"

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
        
    if not flags:
        flags.append("No immediate danger signs detected")
        
    return flags

def predict_disease(symptoms: list, existing_conditions: list = [], 
                    is_pregnant: bool = False, severity: int = 5,
                    recent_travel: bool = False, age: int = 30, 
                    duration_days: int = 1) -> dict:
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
    
    # Convert to DataFrame to avoid feature name warnings
    vec_df = pd.DataFrame(vector, columns=_symptom_columns)
    probs = _model.predict_proba(vec_df)[0]

    # Step 1: PRE-PROCESS ML RESULTS
    top_indices = np.argsort(probs)[::-1][:30] # Get more candidates for filtering
    
    raw_ml_conditions = []
    for idx in top_indices:
        disease_name = _disease_encoder.classes_[idx]
        raw_prob = float(probs[idx])
        
        medicine_info = get_medicine_info(disease_name).copy()
        raw_ml_conditions.append({
            "name": disease_name,
            "raw_probability": raw_prob,
            "medicine_info": medicine_info
        })

    # Step 2: FILTER AGE-INAPPROPRIATE DISEASES
    top_conditions = filter_age_inappropriate_diseases(raw_ml_conditions, age)
    
    # Filter diseases with no symptom match
    top_conditions = [
        c for c in top_conditions
        if has_minimum_symptom_match(c["name"], matched_symptoms)
    ]

    if not top_conditions:
        top_conditions = raw_ml_conditions[:1]

    # Step 3: ENHANCED CONFIDENCE + REASONING (Knowledge Engine)
    enhanced_conditions = []
    # Narrow down to top candidates for enhancement
    for condition in top_conditions[:10]:
        disease_name = condition["name"]
        ml_prob = condition["raw_probability"]
        
        conf = calculate_confidence(
            disease_name, matched_symptoms, age, 
            duration_days, ml_prob
        )
        
        reasoning = build_reasoning(
            disease_name, symptoms, age,
            duration_days, conf
        )
        
        # Medicine safety filter
        medicine_info = condition["medicine_info"]
        restricted_drugs = ["Heparin IV", "Clopidogrel", "Nitroglycerin"]
        if "prescription_medicines" in medicine_info:
            medicine_info["prescription_medicines"] = [
                m for m in medicine_info["prescription_medicines"] 
                if not any(r.lower() in m.lower() for r in restricted_drugs)
            ]
            
        enhanced_conditions.append({
            "name": disease_name,
            "confidence": conf["label"],
            "conf_score": conf["score"], # Store for sorting
            "match_score": conf["match_score"],
            "reasoning": reasoning,
            "medicine_info": {
                **medicine_info,
                "disclaimer": "⚠️ DO NOT SELF-MEDICATE — For awareness only. Always consult a doctor before taking any medicine."
            }
        })

    # Step 4: SORT BY CONFIDENCE + COMMONALITY
    enhanced_conditions.sort(
        key=lambda x: (
            -{"High": 3, "Medium": 2, "Low": 1}.get(x["confidence"], 0),
            DISEASE_COMMONALITY_RANK.get(x["name"], 99)
        )
    )

    # Step 5: GET URGENCY (Knowledge Engine)
    top_disease = enhanced_conditions[0]["name"] if enhanced_conditions else "Unknown"
    urgency, urgency_reason = get_urgency(
        top_disease, matched_symptoms, severity,
        age, duration_days, is_pregnant
    )
    
    # Red flags and recommended tests
    red_flags = get_red_flags(matched_symptoms, existing_conditions)
    top_condition_name = enhanced_conditions[0]["name"]
    recommended_tests = DISEASE_TESTS.get(top_condition_name, ["Consult a doctor for specific tests"])
    
    # Home care advice
    c_lower = top_condition_name.lower()
    if any(k in c_lower for k in ["fungal", "skin", "acne", "psoriasis", "impetigo", "rash"]):
        home_care = ["Keep area clean", "Avoid scratching", "Wear loose clothing"]
    elif any(k in c_lower for k in ["fever", "malaria", "dengue", "typhoid"]):
        home_care = ["Paracetamol for fever", "Stay hydrated", "Rest"]
    elif "cold" in c_lower or "allergy" in c_lower:
        home_care = ["Stay warm", "Steam inhalation", "Avoid allergens"]
    elif any(k in c_lower for k in ["gastro", "jaundice", "diarrhoea"]):
        home_care = ["ORS for hydration", "Avoid solid food for few hours"]
    else:
        home_care = ["Rest and stay hydrated"]
    
    doctor_summary = (f"Patient presents with {', '.join(symptoms)}. "
                      f"AI triage identifies {top_condition_name} as most probable. "
                      f"Urgency level: {urgency}.")
    
    return {
        "urgency": urgency,
        "urgency_reason": urgency_reason,
        "top_conditions": enhanced_conditions[:3],
        "red_flags": red_flags,
        "recommended_tests": recommended_tests,
        "home_care": home_care,
        "when_to_escalate": ["If symptoms worsen significantly", "If new symptoms appear"],
        "doctor_summary": doctor_summary,
        "source": "ml_only"
    }
