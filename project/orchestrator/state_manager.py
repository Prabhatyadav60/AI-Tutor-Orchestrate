from typing import Dict

def get_user_context(user_id: str) -> Dict[str, str]:
    # Stub - in production this would query a DB.
    # Provide fields required by the tools per the hackathon spec.
    return {
        "user_id": user_id,
        "name": "Student",
        "grade_level": "10",
        "learning_style_summary": "Prefers visual explanations and structured notes",
        "emotional_state_summary": "Confused",
        "mastery_level_summary": "Level 4: Building foundational knowledge",
        "preferred_teaching_style": "visual"
    }
