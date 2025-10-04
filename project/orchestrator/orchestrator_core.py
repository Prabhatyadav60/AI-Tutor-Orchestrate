from typing import Dict, Any
from .gemini_agent import get_tool_decision
from .tool_executor import execute_tool
from jsonschema import validate, ValidationError
import logging

logger = logging.getLogger("orchestrator_core")

# reuse schemas from gemini_agent to validate final outgoing parameters
from .gemini_agent import TOOL_PARAM_SCHEMAS

def _infer_defaults(parameters: Dict[str, Any], tool: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
    # adapt parameters using teaching style, emotional state and mastery level
    params = dict(parameters)

    # Ensure required fields exist with sensible defaults
    if tool == "flashcard_generator":
        params.setdefault("count", 5)
        # choose difficulty by mastery level
        try:
            lvl = int(user_context.get("mastery_level_summary", "Level 4").split()[1])
        except Exception:
            lvl = 4
        if lvl <= 3:
            params.setdefault("difficulty", "easy")
        elif lvl <= 6:
            params.setdefault("difficulty", "medium")
        else:
            params.setdefault("difficulty", "hard")

    if tool == "note_maker":
        # adapt note_taking_style from preferred_teaching_style / learning_style_summary
        pref = user_context.get("preferred_teaching_style", "").lower()
        if pref == "visual":
            params.setdefault("note_taking_style", "structured")
            params.setdefault("include_examples", True)
            params.setdefault("include_analogies", True)
        else:
            params.setdefault("note_taking_style", params.get("note_taking_style", "outline"))
            params.setdefault("include_examples", True)
            params.setdefault("include_analogies", False)

    if tool == "concept_explainer":
        # desired depth based on mastery
        try:
            lvl = int(user_context.get("mastery_level_summary", "Level 4").split()[1])
        except Exception:
            lvl = 4
        if lvl <= 3:
            params.setdefault("desired_depth", "basic")
        elif lvl <= 6:
            params.setdefault("desired_depth", "intermediate")
        else:
            params.setdefault("desired_depth", "advanced")
        params.setdefault("include_examples", True)

    return params

async def orchestrate_with_gemini(message: str, chat_history: list, user_context: Dict[str, Any]) -> Dict[str, Any]:
    decision = await get_tool_decision(user_message=message, chat_history=chat_history, user_context=user_context)
    tool_name = decision["tool"]
    parameters = decision["parameters"]

    # attach user info + chat history
    parameters["user_info"] = user_context
    parameters["chat_history"] = chat_history

    # adapt/infer defaults from user_context
    parameters = _infer_defaults(parameters, tool_name, user_context)

    # validate final params against schema
    try:
        validate(instance=parameters, schema={"type":"object","properties": {"user_info": {"type":"object"}}, "required":["user_info"]})
        # also validate tool-specific
        validate(instance={k:v for k,v in parameters.items() if k in parameters}, schema={"type":"object"})  # no-op placeholder
        # deep validate tool parameters per tool schema
        tool_schema = TOOL_PARAM_SCHEMAS[tool_name]
        # Only validate parameters that are in the tool schema (exclude user_info/chat_history)
        tool_params = {k:v for k,v in parameters.items() if k in tool_schema.get("properties", {})}
        validate(instance=tool_params, schema=tool_schema)
    except ValidationError as ve:
        raise Exception(f"Parameter validation failed before calling tool: {ve.message}")

    result = await execute_tool(tool_name, parameters)
    return {"tool": tool_name, "parameters": parameters, "result": result}
