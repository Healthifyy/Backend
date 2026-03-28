from fastapi import APIRouter, HTTPException
from models.schemas import TriageRequest, TriageResponse, ImageSymptomRequest, ImageSymptomResponse
from ml.predict import predict_disease, load_model, validate_symptoms
from ml.gemini_enhancer import enhance_with_gemini
from datetime import datetime
import logging

# Set up logging for this router
logger = logging.getLogger("healthify.triage")

router = APIRouter()

@router.post("/triage", response_model=TriageResponse)
async def triage_patient(request: TriageRequest):
    """
    Main triage entry point:
    1. Runs the ML RandomForest model for structured triage
    2. Enhances the result with Gemini 2.0 Flash for clinical reasoning
    3. Handles fallbacks for model missing or API limits
    """
    try:
        # LOG INCOMING REQUEST
        print(f"[{datetime.now()}] Triage request — Age: {request.age}, Gender: {request.gender}, Symptoms: {request.symptoms}")
        
        # PREPARE DATA FOR ENHANCEMENT
        patient_data = request.model_dump()
        
        # 1. RUN ML PREDICTION (RandomForest)
        try:
            ml_result = predict_disease(
                symptoms=request.symptoms,
                existing_conditions=request.existing_conditions,
                is_pregnant=request.is_pregnant,
                severity=request.severity
            )
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            raise HTTPException(status_code=503, detail="ML model unavailable")

        # 2. RUN GEMINI ENHANCEMENT
        try:
            final_result = enhance_with_gemini(patient_data, ml_result)
        except Exception as e:
            logger.warning(f"Gemini enhancement failed, falling back to ML: {e}")
            final_result = ml_result

        # 3. PERSONALIZE DOCTOR SUMMARY
        if request.name:
            # Replace generic 'Patient presents' with the actual patient name
            final_result["doctor_summary"] = final_result["doctor_summary"].replace(
                "Patient presents", 
                f"{request.name} presents"
            )

        # LOG THE FINAL RESULT
        print(f"[{datetime.now()}] Result — Urgency: {final_result['urgency']}, Top: {final_result['top_conditions'][0]['name']}, Source: {final_result['source']}")
        
        return final_result

    except HTTPException:
        # Re-raise HTTP exceptions (from ML failure)
        raise
    except Exception as e:
        # GLOBAL ERROR FALLBACK — Ensure the system never crashes
        logger.error(f"Unexpected triage error: {e}")
        return {
            "urgency": "routine",
            "urgency_reason": "Analysis unavailable — please consult a doctor",
            "top_conditions": [],
            "red_flags": [],
            "recommended_tests": ["Please consult a doctor"],
            "home_care": ["Please seek medical advice"],
            "when_to_escalate": ["If symptoms worsen significantly"],
            "doctor_summary": "Unable to complete analysis. Please consult a healthcare provider.",
            "source": "error_fallback"
        }

@router.post("/validate-image-symptoms", response_model=ImageSymptomResponse)
async def validate_image_symptoms(request: ImageSymptomRequest):
    """
    Validate symptoms detected from an image against the model's known symptom list.
    """
    try:
        valid = validate_symptoms(request.detected_symptoms)
        return {"valid_symptoms": valid}
    except Exception as e:
        logger.error(f"Symptom validation failed: {e}")
        # Fallback to returning original symptoms if validation fails
        return {"valid_symptoms": request.detected_symptoms}
