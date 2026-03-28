import json
import os

# Load medical knowledge
_knowledge_path = os.path.join(os.path.dirname(__file__), 'medical_knowledge.json')
with open(_knowledge_path, 'r') as f:
    MEDICAL_KNOWLEDGE = json.load(f)['diseases']

def get_age_group(age: int) -> str:
    if age < 18:
        return "child"
    elif age < 60:
        return "adult"
    else:
        return "elderly"

def get_duration_category(duration_days: int) -> str:
    if duration_days <= 3:
        return "acute"
    elif duration_days <= 14:
        return "subacute"
    else:
        return "chronic"

def calculate_confidence(
    disease_name: str,
    input_symptoms: list,
    age: int,
    duration_days: int,
    ml_probability: float
) -> dict:
    """
    Calculate enhanced confidence score using:
    - ML probability
    - Symptom match ratio
    - Age appropriateness
    - Duration appropriateness
    """
    if disease_name not in MEDICAL_KNOWLEDGE:
        # Unknown disease — use ML probability only
        conf_score = ml_probability
        label = _score_to_label(conf_score)
        return {
            "score": round(conf_score, 2),
            "label": label,
            "match_score": round(ml_probability * 100),
            "age_appropriate": True,
            "duration_appropriate": True
        }

    knowledge = MEDICAL_KNOWLEDGE[disease_name]
    age_group = get_age_group(age)
    duration_cat = get_duration_category(duration_days)

    # 1. Symptom match score
    primary = knowledge["primary_symptoms"]
    secondary = knowledge["secondary_symptoms"]
    all_known = primary + secondary

    primary_matches = [s for s in input_symptoms if s in primary]
    secondary_matches = [s for s in input_symptoms if s in secondary]

    if len(all_known) > 0:
        match_ratio = (
            len(primary_matches) * 2 + len(secondary_matches)
        ) / (len(primary) * 2 + len(secondary))
    else:
        match_ratio = 0.5

    # 2. Age appropriateness
    age_weight = knowledge["age_weight"].get(age_group, 1.0)
    age_min = knowledge.get("exclude_if_age_below", 0)
    age_max = knowledge.get("exclude_if_age_above", 100)
    age_appropriate = age_min <= age <= age_max

    if not age_appropriate:
        age_weight = 0.1  # heavily penalize

    # 3. Duration appropriateness
    duration_weight = knowledge["duration_weight"].get(duration_cat, 1.0)
    min_dur = knowledge.get("min_duration_days", 1)
    duration_appropriate = duration_days >= min_dur

    if not duration_appropriate:
        duration_weight *= 0.5

    # 4. Final score combining all factors
    boost = knowledge.get("confidence_boost", 0.2)
    conf_score = (
        ml_probability * 0.15 +
        match_ratio * 0.50 +
        boost * 0.15 +
        (age_weight - 1.0) * 0.10 +
        (duration_weight - 1.0) * 0.10
    )
    conf_score = min(0.98, max(0.05, conf_score))

    # 5. Match score as percentage
    match_pct = round(
        (len(primary_matches) * 2 + len(secondary_matches)) /
        max(1, len(primary) * 2 + len(secondary)) * 100
    )

    return {
        "score": round(conf_score, 2),
        "label": _score_to_label(conf_score),
        "match_score": match_pct,
        "age_appropriate": age_appropriate,
        "duration_appropriate": duration_appropriate,
        "primary_matches": primary_matches,
        "secondary_matches": secondary_matches
    }

def _score_to_label(score: float) -> str:
    if score >= 0.70:
        return "High"
    elif score >= 0.45:
        return "Medium"
    else:
        return "Low"

def get_urgency(
    disease_name: str,
    input_symptoms: list,
    severity: int,
    age: int,
    duration_days: int,
    is_pregnant: bool
) -> tuple:
    """
    Returns (urgency_level, urgency_reason)
    Levels: EMERGENCY, URGENT, MODERATE, NON_URGENT
    """
    age_group = get_age_group(age)
    
    # Hard emergency rules — always override
    emergency_symptoms = [
        "chest_pain", "breathlessness", "altered_sensorium",
        "weakness_in_limbs", "slurred_speech", "loss_of_consciousness"
    ]
    emergency_matches = [s for s in input_symptoms if s in emergency_symptoms]
    
    if len(emergency_matches) >= 2 or (
        "chest_pain" in input_symptoms and severity >= 7
    ):
        return "EMERGENCY", f"Critical symptoms detected: {', '.join(emergency_matches)}"

    if disease_name in MEDICAL_KNOWLEDGE:
        base_urgency = MEDICAL_KNOWLEDGE[disease_name]["urgency_base"]
        red_flags = MEDICAL_KNOWLEDGE[disease_name]["red_flags"]
        red_flag_matches = [s for s in input_symptoms if s in red_flags]
    else:
        base_urgency = "MODERATE"
        red_flag_matches = []

    # Escalation rules
    if base_urgency == "EMERGENCY":
        return "EMERGENCY", "Condition requires immediate emergency care"

    if red_flag_matches and severity >= 6:
        if base_urgency in ["MODERATE", "NON_URGENT"]:
            base_urgency = "URGENT"

    # Age escalation
    if age_group == "elderly" and base_urgency == "MODERATE":
        base_urgency = "URGENT"
        return base_urgency, "Elevated urgency due to patient age"

    if age_group == "child" and age < 5 and severity >= 5:
        if base_urgency == "NON_URGENT":
            base_urgency = "MODERATE"

    # Pregnancy escalation
    if is_pregnant and base_urgency in ["NON_URGENT", "MODERATE"]:
        base_urgency = "URGENT"
        return base_urgency, "Elevated urgency due to pregnancy"

    # Severity-based adjustment
    if severity >= 8 and base_urgency == "MODERATE":
        base_urgency = "URGENT"
    elif severity >= 6 and base_urgency == "NON_URGENT":
        base_urgency = "MODERATE"
    elif severity <= 3 and base_urgency == "MODERATE":
        base_urgency = "NON_URGENT"
    elif severity <= 4 and base_urgency == "URGENT":
        base_urgency = "MODERATE"

    # Duration-based adjustment
    duration_cat = get_duration_category(duration_days)
    if duration_cat == "chronic" and base_urgency == "NON_URGENT":
        base_urgency = "MODERATE"

    reasons = {
        "EMERGENCY": "Requires immediate emergency care — call 108",
        "URGENT": "Needs medical attention within 24 hours",
        "MODERATE": "Consult a doctor within 2-3 days",
        "NON_URGENT": "Monitor symptoms, consult doctor if no improvement in 5 days"
    }

    return base_urgency, reasons.get(base_urgency, "Consult a doctor")

def filter_age_inappropriate_diseases(
    conditions: list,
    age: int
) -> list:
    """Remove diseases that are age-inappropriate"""
    filtered = []
    for condition in conditions:
        name = condition.get("name", "")
        if name in MEDICAL_KNOWLEDGE:
            min_age = MEDICAL_KNOWLEDGE[name].get("exclude_if_age_below", 0)
            max_age = MEDICAL_KNOWLEDGE[name].get("exclude_if_age_above", 100)
            if min_age <= age <= max_age:
                filtered.append(condition)
            else:
                # Skip this disease — age inappropriate
                continue
        else:
            filtered.append(condition)
    return filtered

def build_reasoning(
    disease_name: str,
    input_symptoms: list,
    age: int,
    duration_days: int,
    confidence_result: dict
) -> str:
    """Build human-readable reasoning string"""
    age_group = get_age_group(age)
    duration_cat = get_duration_category(duration_days)
    primary_matches = confidence_result.get("primary_matches", [])
    
    parts = []
    if primary_matches:
        readable = [s.replace("_", " ") for s in primary_matches]
        parts.append(f"Suspected due to: {', '.join(readable)}")
    
    if not confidence_result.get("age_appropriate", True):
        parts.append(f"Note: Less typical for {age_group} age group")
    
    if not confidence_result.get("duration_appropriate", True):
        parts.append("Note: Duration shorter than typical for this condition")

    label = confidence_result.get("label", "Low")
    parts.append(f"{label} confidence")

    return " — ".join(parts) if parts else "Matched based on symptom pattern"
