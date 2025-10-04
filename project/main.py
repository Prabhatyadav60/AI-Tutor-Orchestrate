import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from orchestrator.orchestrator_core import orchestrate_with_gemini
from orchestrator.state_manager import get_user_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

class ChatMessage(BaseModel):
    role: str
    content: str

class OrchestrateRequest(BaseModel):
    user_id: str = Field(..., example="student123")
    message: str = Field(..., example="I need practice problems on derivatives")
    chat_history: Optional[List[ChatMessage]] = []

app = FastAPI(title="Dynamic AI Tutor Orchestrator (Gemini-driven)")

@app.post("/orchestrate/", summary="Main orchestration endpoint")
async def orchestrate_endpoint(payload: OrchestrateRequest):
    try:
        user_context = get_user_context(payload.user_id)
        result = await orchestrate_with_gemini(
            message=payload.message,
            chat_history=[m.dict() for m in (payload.chat_history or [])],
            user_context=user_context
        )
        return {"ok": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Orchestration failed")
        raise HTTPException(status_code=500, detail=str(e))





if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
