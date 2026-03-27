from ml.gemini_enhancer import enhance_with_gemini
import json

patient_data = {
    "name": "Ramesh Kumar",
    "age": 45,
    "gender": "male", 
    "symptoms": ["chest_pain", "high_fever", "breathlessness", "cough"],
    "duration_days": 5,
    "severity": 8,
    "existing_conditions": ["diabetes"],
    "is_pregnant": False,
    "recent_travel": False
}

ml_result = {
    "urgency": "emergency",
    "urgency_reason": "Emergency symptom detected: chest_pain",
    "top_conditions": [
        {"name": "Heart attack", "confidence": "medium", 
         "match_score": 55, "reasoning": "Matched by symptom pattern"},
        {"name": "Pneumonia", "confidence": "low", 
         "match_score": 14, "reasoning": "Matched by symptom pattern"},
        {"name": "Bronchial Asthma", "confidence": "low", 
         "match_score": 16, "reasoning": "Matched by symptom pattern"}
    ],
    "red_flags": ["Chest pain with breathlessness — possible cardiac emergency"],
    "recommended_tests": ["ECG", "Troponin", "CBC"],
    "home_care": ["Rest and stay hydrated"],
    "when_to_escalate": ["If symptoms worsen"],
    "doctor_summary": "Patient presents with emergency symptoms.",
    "source": "ml_only"
}

def main():
    print("Testing Gemini Enhancement (GenAI SDK)...")
    result = enhance_with_gemini(patient_data, ml_result)
    
    if result['source'] == "ml+gemini":
        print("SUCCESS - source: ml+gemini")
    elif result['source'] == "ml_only":
        print("FALLBACK - source: ml_only — check API key in .env")
    
    print("\n--- FULL JSON RESULT ---")
    print(json.dumps(result, indent=2))
    print(f"\nFinal Source: {result['source']}")

if __name__ == "__main__":
    main()
