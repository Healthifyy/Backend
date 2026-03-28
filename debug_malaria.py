import ml.predict
import numpy as np
from ml.knowledge_engine import has_minimum_symptom_match

ml.predict.load_model()
symptoms = ['high_fever','chills','sweating','headache']
vector = ml.predict.symptoms_to_vector(symptoms)
probs = ml.predict._model.predict_proba(vector)[0]
top_indices = np.argsort(probs)[::-1][:20]

print("Top 20 ML results for Test 3:")
found_malaria = False
for idx in top_indices:
    name = ml.predict._disease_encoder.classes_[idx]
    has_match = has_minimum_symptom_match(name, symptoms)
    print(f"  {name}: {probs[idx]:.4f} | Match: {has_match}")
    if name == "Malaria":
        found_malaria = True

if not found_malaria:
    try:
        malaria_idx = list(ml.predict._disease_encoder.classes_).index("Malaria")
        print(f"\nMalaria NOT in top 20! Prob: {probs[malaria_idx]:.4f}")
    except ValueError:
        print("\nMalaria NOT FOUND in classes!")
