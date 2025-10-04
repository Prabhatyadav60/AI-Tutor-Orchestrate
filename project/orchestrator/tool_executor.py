import httpx
import asyncio
import logging
from typing import Dict, Any
from .config import HTTP_TIMEOUT_SECONDS

logger = logging.getLogger("tool_executor")

TOOL_ENDPOINTS = {
    "note_maker": "http://localhost:8101/note_maker",
    "flashcard_generator": "http://localhost:8102/flashcards",
    "concept_explainer": "http://localhost:8103/explain"
}

async def execute_tool(tool_name: str, parameters: Dict[str, Any], timeout: int = HTTP_TIMEOUT_SECONDS) -> Dict[str, Any]:
    endpoint = TOOL_ENDPOINTS.get(tool_name)
    if not endpoint:
        raise Exception(f"No endpoint configured for tool '{tool_name}'")

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(endpoint, json=parameters)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Tool call failed for {tool_name}: {e}. Returning mock response.")
            await asyncio.sleep(0.05)
            return _mock_response(tool_name, parameters)

def _mock_response(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "note_maker":
        return {
            "topic": params.get("topic", "general"),
            "title": f"Notes: {params.get('topic','general')}",
            "summary": f"Auto-generated notes for {params.get('topic','general')}.",
            "note_sections": [
                {"title": "Key idea", "content": "Concept explained", "key_points": ["p1","p2"], "examples": []}
            ],
            "note_taking_style": params.get("note_taking_style", "structured")
        }
    if tool_name == "flashcard_generator":
        count = int(params.get("count", 5))
        return {
            "topic": params.get("topic", "general"),
            "flashcards": [
                {"title": f"{params.get('topic','Q')} #{i+1}", "question": f"Q{i+1}", "answer": f"A{i+1}", "example": ""}
                for i in range(min(20, count))
            ],
            "difficulty": params.get("difficulty", "medium"),
            "adaptation_details": "mocked"
        }
    if tool_name == "concept_explainer":
        return {
            "explanation": f"A concise explanation of {params.get('concept_to_explain','the concept')}.",
            "examples": ["Example 1", "Example 2"],
            "related_concepts": ["related 1", "related 2"],
            "practice_questions": ["Q1", "Q2"]
        }
    return {"result": "unknown tool"}
