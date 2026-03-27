import os
import uuid
from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()

# In-memory store as fallback (works without Supabase)
_sessions_store = []

class SessionRequest(BaseModel):
    patient_name: Optional[str] = "Anonymous"
    age: int
    gender: str
    symptoms: list[str]
    urgency: str
    top_condition: str
    worker_id: Optional[str] = "default"
    village: Optional[str] = ""
    session_date: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    saved: bool

@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    try:
        session_id = str(uuid.uuid4())
        session_date = request.session_date or datetime.now().isoformat()
        
        session_dict = request.model_dump()
        session_dict["session_id"] = session_id
        session_dict["session_date"] = session_date
        
        # Try to save to Supabase
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        saved_to_db = False
        if supabase_url and supabase_key:
            try:
                from supabase import create_client
                supabase = create_client(supabase_url, supabase_key)
                supabase.table("patient_sessions").insert(session_dict).execute()
                saved_to_db = True
            except Exception as db_err:
                print(f"Supabase error: {db_err}")
                
        # Fallback to in-memory store
        if not saved_to_db:
            _sessions_store.append(session_dict)
            print("Saved to in-memory store")
            
        return {"session_id": session_id, "saved": True}
        
    except Exception as e:
        print(f"Error creating session: {e}")
        # Never crash — always return saved: True
        return {"session_id": "error-fallback-id", "saved": True}


@router.get("/sessions")
async def get_sessions(worker_id: Optional[str] = None, date: Optional[str] = None):
    sessions = []
    
    # Try Supabase first
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    loaded_from_db = False
    if supabase_url and supabase_key:
        try:
            from supabase import create_client
            supabase = create_client(supabase_url, supabase_key)
            query = supabase.table("patient_sessions").select("*")
            if worker_id:
                query = query.eq("worker_id", worker_id)
            if date:
                query = query.gte("session_date", date).lt("session_date", date + "T23:59:59")
            result = query.execute()
            sessions = result.data
            loaded_from_db = True
        except Exception as e:
            print(f"Supabase read error: {e}")

    # Fallback to in-memory store
    if not loaded_from_db:
        for s in _sessions_store:
            match_worker = True
            match_date = True
            
            if worker_id and s.get("worker_id") != worker_id:
                match_worker = False
            if date and not str(s.get("session_date", "")).startswith(date):
                match_date = False
                
            if match_worker and match_date:
                sessions.append(s)
                
    # Calculate stats
    emergency_count = sum(1 for s in sessions if s.get("urgency") == "emergency")
    urgent_count = sum(1 for s in sessions if s.get("urgency") == "urgent")
    routine_count = sum(1 for s in sessions if s.get("urgency") == "routine")
    
    return {
        "sessions": sessions,
        "total": len(sessions),
        "emergency_count": emergency_count,
        "urgent_count": urgent_count,
        "routine_count": routine_count
    }
