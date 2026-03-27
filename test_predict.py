from ml.predict import predict_disease
import json

symptoms = ['chest_pain', 'high_fever', 'breathlessness']
result = predict_disease(symptoms)
print(json.dumps(result, indent=2))
