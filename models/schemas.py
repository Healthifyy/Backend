from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class TriageRequest(BaseModel):
    """Patient symptom intake for AI triage."""

    name: Optional[str] = Field(default=None, max_length=100)
    age: int = Field(..., ge=1, le=120, description="Patient age in years")
    gender: Literal["male", "female", "other"]
    symptoms: List[str] = Field(..., min_length=1)
    duration_days: int = Field(..., ge=1, le=365)
    severity: int = Field(..., ge=1, le=10)
    existing_conditions: List[str] = Field(default=[])
    medications: str = Field(default="")
    is_pregnant: bool = Field(default=False)
    recent_travel: bool = Field(default=False)
    community_outbreak: bool = Field(default=False)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Ramesh Kumar",
                "age": 45,
                "gender": "male",
                "symptoms": ["chest_pain", "high_fever", "breathlessness"],
                "duration_days": 5,
                "severity": 8,
                "existing_conditions": ["diabetes"],
                "medications": "Metformin 500mg",
                "is_pregnant": False,
                "recent_travel": False,
                "community_outbreak": False,
            }
        }


class ConditionResult(BaseModel):
    """A single predicted condition with confidence and reasoning."""

    name: str
    confidence: str  # "high" | "medium" | "low"
    match_score: int
    reasoning: str


class TriageResponse(BaseModel):
    """Full triage result returned to the frontend."""

    urgency: str  # "emergency" | "urgent" | "routine"
    urgency_reason: str
    top_conditions: List[ConditionResult]
    red_flags: List[str]
    recommended_tests: List[str]
    home_care: List[str]
    when_to_escalate: List[str]
    doctor_summary: str
    source: str = "ml_only"  # "ml+gemini" | "ml_only"
