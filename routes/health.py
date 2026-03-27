from fastapi import APIRouter
from ml import predict
from routes.sessions import _sessions_store
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint — frontend polls this on load.
    Reflects the actual current state of the ML model.
    """
    return {
        "status": "ok",
        "model_loaded": predict._model is not None,
        "gemini_available": False,  # Note: Gemini is verified on-the-fly via rotate logic
        "version": "1.0.0",
        "timestamp": predict.datetime.now().isoformat() if hasattr(predict, 'datetime') else None
    }


@router.get("/stats")
async def stats():
    """
    Stats endpoint for dashboard usage.
    """
    today = datetime.now().isoformat()[:10]
    # Count sessions that occurred today
    today_sessions = [s for s in _sessions_store if str(s.get("session_date", "")).startswith(today)]
    
    return {
        "status": "ok",
        "model_loaded": predict._model is not None,
        "total_triages_today": len(today_sessions),
        "version": "1.0.0"
    }
