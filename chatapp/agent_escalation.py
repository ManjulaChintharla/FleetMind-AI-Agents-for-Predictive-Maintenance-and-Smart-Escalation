import os
import requests
from dotenv import load_dotenv
load_dotenv()


# Load Logic App URL from .env
LOGIC_APP_URL = os.getenv("LOGICAPP_TRIGGER_URL")

def evaluate_escalation(question: str, context: str) -> str:
    """
    Trigger escalation only for critical, safety-related issues.
    """
    urgent_keywords = [
        "fire", "fuel leak", "explosion", "overheat", "brake failure", "engine shut down",
        "burning smell", "black smoke", "power loss", "engine failure", "stall", "vibrating badly"
    ]
    combined = f"{question} {context}".lower()
    return "ESCALATE" if any(k in combined for k in urgent_keywords) else "NO_ACTION"

def trigger_logic_app(issue: str, context: str) -> bool:
    try:
        payload = {
            "issue": issue,
            "context": context
        }
        response = requests.post(
            LOGIC_APP_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        return response.status_code in [200, 202]
    except Exception as e:
        print("Logic App escalation error:", str(e))
        return False
