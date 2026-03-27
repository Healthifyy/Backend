import sys
import json
import warnings
warnings.filterwarnings('ignore')

from ml.predict import predict_disease

tests = [
    {
        "desc": "Test 1",
        "symptoms": ["chest_pain", "high_fever", "breathlessness", "cough"],
        "severity": 8, "age": 45, "gender": "male", "existing_conditions": ["diabetes"],
        "is_pregnant": False, "recent_travel": False
    },
    {
        "desc": "Test 2",
        "symptoms": ["high_fever", "headache", "body_ache"],
        "severity": 6, "age": 28, "gender": "female", "existing_conditions": [],
        "is_pregnant": True, "recent_travel": False
    },
    {
        "desc": "Test 3",
        "symptoms": ["runny_nose", "sneezing", "mild_fever"],
        "severity": 3, "age": 8, "gender": "male", "existing_conditions": [],
        "is_pregnant": False, "recent_travel": False
    }
]

for t in tests:
    print(f"\n--- {t['desc']} ---")
    res = predict_disease(
        symptoms=t["symptoms"],
        existing_conditions=t["existing_conditions"],
        is_pregnant=t["is_pregnant"],
        severity=t["severity"],
        recent_travel=t["recent_travel"],
        age=t["age"],
        gender=t["gender"]
    )
    print(json.dumps(res, indent=2))
