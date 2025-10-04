import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = os.getenv(
    "GEMINI_API_URL",
    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
)

HEADERS = {
    "Content-Type": "application/json"
}

# timeouts / retry policy tweakable here
HTTP_TIMEOUT_SECONDS = int(os.getenv("HTTP_TIMEOUT_SECONDS", "20"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
