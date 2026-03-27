from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routes.health import router as health_router
from routes.triage import router as triage_router
from routes.sessions import router as sessions_router
from utils.logger import setup_logger
from ml.predict import load_model

# Load environment variables from .env
load_dotenv()

logger = setup_logger("healthify")

app = FastAPI(
    title="Healthify API",
    description="AI-powered health triage for rural Indian communities",
    version="1.0.0",
)

# CORS — allow all origins (hackathon mode)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        logger.info("ML Prediction model artifacts loaded successfully")
        print("ML Prediction model artifacts loaded successfully")
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
