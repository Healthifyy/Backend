from ml.knowledge_engine import has_minimum_symptom_match, MEDICAL_KNOWLEDGE

symptoms = ['continuous_sneezing', 'runny_nose', 'mild_fever', 'cough']
# Verify Hepatitis A symptoms in knowledge
hepa = MEDICAL_KNOWLEDGE["Hepatitis A"]
print("Hepatitis A Primary:", hepa["primary_symptoms"])
print("Hepatitis A Secondary:", hepa["secondary_symptoms"])

match = has_minimum_symptom_match("Hepatitis A", symptoms)
print("\nSymptom match for Hepatitis A:", match)

# Check Common Cold
cold = MEDICAL_KNOWLEDGE["Common Cold"]
print("\nCommon Cold Primary:", cold["primary_symptoms"])
print("Common Cold Secondary:", cold["secondary_symptoms"])
match_cold = has_minimum_symptom_match("Common Cold", symptoms)
print("Symptom match for Common Cold:", match_cold)
