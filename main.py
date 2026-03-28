from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routes.health import router as health_router
from routes.triage import router as triage_router
from routes.sessions import router as sessions_router
from utils.logger import setup_logger
from ml.predict import load_model, get_symptom_list
from models.schemas import ImageSymptomRequest

# Global cache for symptom names (loaded at startup)
SYMPTOM_LIST = []

# Load environment variables from .env
load_dotenv()

logger = setup_logger("healthify")

app = FastAPI(
    title="Healthify API",
    description="AI-powered health triage for rural Indian communities",
    version="1.0.0",
)

# CORS — allow all origins for local + Vercel
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://healthify-frontend-lake.vercel.app",
    "https://*.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="", tags=["health"])
app.include_router(triage_router, prefix="", tags=["triage"])
app.include_router(sessions_router, prefix="", tags=["sessions"])


@app.on_event("startup")
async def startup_event():
    """Startup event handler: loads model and logs stats."""
    logger.info("Healthify API starting up...")
    print("Healthify API starting up...")
    
    # LOAD ML MODEL
    success = load_model()
    if success:
        global SYMPTOM_LIST
        SYMPTOM_LIST = get_symptom_list()
        logger.info(f"ML Prediction model artifacts loaded successfully ({len(SYMPTOM_LIST)} symptoms)")
        print(f"ML Prediction model artifacts loaded successfully ({len(SYMPTOM_LIST)} symptoms)")
    else:
        logger.error("Failed to load ML model artifacts. Predictive triage will fail.")
        print("CRITICAL: Failed to load ML model artifacts.")


@app.get("/")
async def root():
    """Root endpoint — confirms the API is live."""
    return {
        "message": "Healthify API",
        "docs": "/docs",
    }


@app.post("/validate-image-symptoms")
async def validate_image_symptoms(request: ImageSymptomRequest):
    """
    Validate symptoms detected from an image against our model's known symptom list.
    Safely handles cases where the model hasn't loaded yet.
    """
    # SAFETY CHECK: If model not loaded/empty, pass all through as valid
    if not SYMPTOM_LIST:
        return {
            "valid_symptoms": request.detected_symptoms,
            "invalid_symptoms": [],
            "confidence": request.confidence,
            "message": f"{len(request.detected_symptoms)} symptoms passed through (model loading)"
        }
    
    valid = []
    invalid = []
    
    for s in request.detected_symptoms:
        # Standardize formatting to match ML column names (lowercase, underscores)
        normalized = s.lower().replace(" ", "_").replace("-", "_")
        
        if normalized in SYMPTOM_LIST:
            valid.append(normalized)
        else:
            invalid.append(s)
    
    return {
        "valid_symptoms": valid,
        "invalid_symptoms": invalid,
        "confidence": request.confidence,
        "message": f"{len(valid)} of {len(request.detected_symptoms)} image symptoms matched our database"
    }
