# orchestrator/gemini_agent.py
import httpx
import json
import re
from typing import Dict, Any, Iterable, List, Set
from jsonschema import validate, ValidationError
from .config import GEMINI_API_URL, HEADERS, HTTP_TIMEOUT_SECONDS, MAX_RETRIES

SYSTEM_INSTRUCTION = """
You are an autonomous orchestration LLM for an AI tutoring middleware.
YOU MUST OUTPUT ONLY VALID JSON (no extra explanation or commentary).
The JSON MUST contain top-level keys:
- tool: one of ["note_maker","flashcard_generator","concept_explainer"]
- parameters: object matching that tool's required input schema

Return exactly one JSON object.

Examples (JSON only):
{ "tool": "note_maker", "parameters": { "topic": "derivatives", "subject": "calculus", "note_taking_style": "outline", "include_examples": true, "include_analogies": false } }
"""

# minimal schemas used to validate LLM output (keeps orchestrator safe)
TOOL_PARAM_SCHEMAS = {
    "note_maker": {
        "type": "object",
        "required": ["topic", "subject", "note_taking_style"],
        "properties": {
            "topic": {"type": "string"},
            "subject": {"type": "string"},
            "note_taking_style": {"type": "string", "enum": ["outline","bullet_points","narrative","structured"]},
            "include_examples": {"type": "boolean"},
            "include_analogies": {"type": "boolean"}
        }
    },
    "flashcard_generator": {
        "type": "object",
        "required": ["topic", "count", "difficulty", "subject"],
        "properties": {
            "topic": {"type": "string"},
            "count": {"type": "integer", "minimum": 1, "maximum": 20},
            "difficulty": {"type": "string", "enum": ["easy","medium","hard"]},
            "include_examples": {"type": "boolean"},
            "subject": {"type": "string"}
        }
    },
    "concept_explainer": {
        "type": "object",
        "required": ["concept_to_explain","current_topic","desired_depth"],
        "properties": {
            "concept_to_explain": {"type": "string"},
            "current_topic": {"type": "string"},
            "desired_depth": {"type": "string", "enum": ["basic","intermediate","advanced","comprehensive"]},
            "include_examples": {"type": "boolean"}
        }
    }
}

async def ask_gemini(prompt: str, timeout: int = HTTP_TIMEOUT_SECONDS) -> str:
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    async with httpx.AsyncClient(timeout=timeout) as client:
        last_exc = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                r = await client.post(GEMINI_API_URL, json=body, headers=HEADERS)
                r.raise_for_status()
                data = r.json()
                # typical path: candidates[0].content.parts[0].text
                return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", json.dumps(data))
            except Exception as e:
                last_exc = e
        raise Exception(f"Gemini API error after retries: {last_exc}")

def _extract_first_json(text: str):
    # robustly find the first balanced JSON object in text
    start = None
    stack = []
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        # continue searching
                        start = None
    raise ValueError("No valid JSON object found in LLM response.")

def _extract_candidates_for_concept(message: str) -> List[str]:
    """
    Heuristic extraction of likely concepts/topics from a free text message.
    Returns list of candidates (most likely first).
    """
    msg = message.strip()
    # common patterns: "explain X", "what is X", "tell me about X", "struggling with X", "practice problems on X"
    patterns = [
        r"(?:explain|explain\s+about)\s+([A-Za-z0-9 _\-/]{2,})",
        r"what(?:'s| is| are)\s+([A-Za-z0-9 _\-/]{2,})\??",
        r"tell me about\s+([A-Za-z0-9 _\-/]{2,})",
        r"struggling with\s+([A-Za-z0-9 _\-/]{2,})",
        r"practice problems (?:on|for)\s+([A-Za-z0-9 _\-/]{2,})",
        r"need (?:help|practice) (?:with|on)\s+([A-Za-z0-9 _\-/]{2,})"
    ]
    candidates = []
    lower = msg.lower()
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            cand = m.group(1).strip()
            # remove trailing punctuation
            cand = re.sub(r"[\.?,!]+$", "", cand).strip()
            if cand:
                candidates.append(cand)
    # fallback: if message short and likely topic-like, use entire message
    if not candidates:
        if len(msg.split()) <= 6:
            candidates.append(msg)
        else:
            # try to take noun-ish phrase: first 6 words
            candidates.append(" ".join(msg.split()[:6]))
    # dedupe
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def _get_mastery_level_int(user_context: Dict[str, Any]) -> int:
    s = user_context.get("mastery_level_summary", "")
    # Expect format like "Level 4: ..." — find first integer
    m = re.search(r"(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return 4  # default reasonable midpoint

def _infer_missing_parameters(tool: str, missing: Set[str], user_message: str, chat_history: list, user_context: dict, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fill missing required parameters with safe defaults inferred from message/context.
    This is intentionally simple/deterministic (no more LLM calls).
    """
    filled = dict(params)  # copy

    # common candidate topics from message
    candidates = _extract_candidates_for_concept(user_message)

    if tool == "concept_explainer":
        for key in missing:
            if key == "concept_to_explain":
                # choose first candidate
                filled["concept_to_explain"] = candidates[0] if candidates else user_message
            elif key == "current_topic":
                # try to reuse concept or use a short form
                filled["current_topic"] = filled.get("concept_to_explain", candidates[0] if candidates else user_message)
            elif key == "desired_depth":
                lvl = _get_mastery_level_int(user_context)
                if lvl <= 3:
                    filled["desired_depth"] = "basic"
                elif lvl <= 6:
                    filled["desired_depth"] = "intermediate"
                else:
                    filled["desired_depth"] = "advanced"
            elif key == "include_examples":
                filled["include_examples"] = True

    elif tool == "flashcard_generator":
        for key in missing:
            if key == "topic":
                filled["topic"] = candidates[0] if candidates else user_message
            elif key == "count":
                filled["count"] = 5
            elif key == "difficulty":
                lvl = _get_mastery_level_int(user_context)
                if lvl <= 3:
                    filled["difficulty"] = "easy"
                elif lvl <= 6:
                    filled["difficulty"] = "medium"
                else:
                    filled["difficulty"] = "hard"
            elif key == "subject":
                # fallback to short candidate or 'general'
                filled["subject"] = candidates[0] if candidates else "general"
            elif key == "include_examples":
                filled["include_examples"] = True

    elif tool == "note_maker":
        for key in missing:
            if key == "topic":
                filled["topic"] = candidates[0] if candidates else user_message
            elif key == "subject":
                filled["subject"] = candidates[0] if candidates else "general"
            elif key == "note_taking_style":
                # prefer structured for visual learners or outline otherwise
                pref = user_context.get("preferred_teaching_style", "").lower()
                if pref == "visual":
                    filled["note_taking_style"] = "structured"
                else:
                    filled["note_taking_style"] = "outline"
            elif key == "include_examples":
                filled["include_examples"] = True
            elif key == "include_analogies":
                pref = user_context.get("preferred_teaching_style", "").lower()
                filled["include_analogies"] = pref == "visual"

    # final safety: ensure strings are strings
    for k, v in list(filled.items()):
        if v is None:
            # replace None with a safe fallback string or boolean
            schema = TOOL_PARAM_SCHEMAS.get(tool, {}).get("properties", {}).get(k, {})
            if schema.get("type") == "string":
                filled[k] = user_message if isinstance(user_message, str) else "general"
            elif schema.get("type") == "boolean":
                filled[k] = True
            elif schema.get("type") == "integer":
                filled[k] = 1

    return filled

async def get_tool_decision(user_message: str, chat_history: list, user_context: dict) -> Dict[str, Any]:
    prompt = SYSTEM_INSTRUCTION + "\n\n"
    prompt += f"Conversation message: {user_message}\n"
    prompt += f"Chat history: {json.dumps(chat_history)}\n"
    prompt += f"User profile: {json.dumps(user_context)}\n\n"
    prompt += "RETURN VALID JSON ONLY."

    raw = await ask_gemini(prompt)
    # extract and validate
    try:
        parsed = _extract_first_json(raw)
    except Exception as e:
        raise Exception(f"Failed to extract JSON from Gemini output. Raw output:\n{raw}\nError: {e}")

    if not isinstance(parsed, dict) or "tool" not in parsed or "parameters" not in parsed:
        raise Exception(f"Gemini JSON missing required keys. Parsed: {parsed}\nRaw output: {raw}")

    tool = parsed["tool"]
    params = parsed["parameters"]
    if tool not in TOOL_PARAM_SCHEMAS:
        raise Exception(f"Gemini suggested unknown tool '{tool}' -- raw: {parsed}")

    schema = TOOL_PARAM_SCHEMAS[tool]
    # quick check: find missing required keys
    reqs: Set[str] = set(schema.get("required", []))
    present: Set[str] = set(k for k, _ in params.items())
    missing = reqs - present

    if missing:
        # attempt to infer missing parameters deterministically
        inferred = _infer_missing_parameters(tool=tool, missing=missing, user_message=user_message, chat_history=chat_history, user_context=user_context, params=params)
        # validate inferred params
        try:
            validate(instance=inferred, schema=schema)
            return {"tool": tool, "parameters": inferred}
        except ValidationError as ve:
            # if inference didn't produce valid params, return detailed error
            raise Exception(
                f"Gemini parameters failed validation and inference couldn't recover. "
                f"Missing: {sorted(missing)}. Validation error: {ve.message}. "
                f"Gemini raw output: {raw}"
            )

    # if nothing missing, validate original params
    try:
        validate(instance=params, schema=schema)
    except ValidationError as ve:
        raise Exception(f"Gemini parameters failed validation: {ve.message}. Raw output: {raw}")

    return {"tool": tool, "parameters": params}
