import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ── Load dataset ──────────────────────────────────────
df = pd.read_csv('ml/datasets/dataset.csv')
df.columns = df.columns.str.strip()

# ── Convert WIDE → BINARY MATRIX ─────────────────────
symptom_cols = [c for c in df.columns if c.startswith('Symptom')]

all_symptoms = set()
for col in symptom_cols:
    vals = df[col].dropna().str.strip().str.lower().str.replace(' ', '_')
    all_symptoms.update(vals.unique())
all_symptoms.discard('nan')
all_symptoms.discard('')
all_symptoms = sorted(list(all_symptoms))

print(f"Unique symptoms found: {len(all_symptoms)}")

# Build binary matrix
binary_rows = []
for _, row in df.iterrows():
    vec = {s: 0 for s in all_symptoms}
    for col in symptom_cols:
        val = row[col]
        if pd.notna(val):
            cleaned = str(val).strip().lower().replace(' ', '_')
            if cleaned in vec:
                vec[cleaned] = 1
    vec['Disease'] = str(row['Disease']).strip()
    binary_rows.append(vec)

binary_df = pd.DataFrame(binary_rows)
print(f"Binary matrix shape: {binary_df.shape}")

# ── Prepare X, y ──────────────────────────────────────
X = binary_df.drop('Disease', axis=1)
y = binary_df['Disease']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print(f"Diseases: {len(encoder.classes_)}")
print(f"Symptom features: {X.shape[1]}")

# ── Stratified train/test split ───────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ── Train Random Forest (depth-limited) ──────────────
print("\nTraining Random Forest (max_depth=15)...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"RF Test Accuracy: {rf_acc:.4f}")

# ── Train Gradient Boosting ───────────────────────────
print("\nTraining Gradient Boosting (max_depth=8)...")
gb = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=8,
    learning_rate=0.1,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
print(f"GB Test Accuracy: {gb_acc:.4f}")

# ── Pick best model ───────────────────────────────────
if gb_acc >= rf_acc:
    best_model = gb
    model_type = "GradientBoosting"
    best_acc = gb_acc
else:
    best_model = rf
    model_type = "RandomForest"
    best_acc = rf_acc

print(f"\nBest model: {model_type} — Accuracy: {best_acc:.4f}")

# ── Classification report ─────────────────────────────
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=encoder.classes_,
    zero_division=0
))

# ── 5-fold cross validation ───────────────────────────
print("\nRunning 5-fold cross validation (takes ~1 min)...")
cv = cross_val_score(best_model, X, y_encoded, cv=5, n_jobs=-1)
print(f"CV Accuracy: {cv.mean():.4f} (+/- {cv.std()*2:.4f})")

# ── Save artifacts ────────────────────────────────────
symptom_columns = list(X.columns)

with open('ml/healthify_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('ml/symptom_columns.pkl', 'wb') as f:
    pickle.dump(symptom_columns, f)

with open('ml/disease_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print(f"\n[OK] Saved: {model_type}")
print(f"[OK] Symptoms: {len(symptom_columns)}")
print(f"[OK] Diseases: {len(encoder.classes_)}")
print("\nDone. Model ready.")
