"""
Free Model Availability Validator
Checks configured model IDs against OpenRouter's live free-tier roster.
"""
import requests
import logging
from typing import List, Dict, Any

from src.config import (
    OPENROUTER_BASE_URL,
    ORCHESTRATOR_MODEL,
    RETRIEVAL_MODEL,
    CRITIC_MODEL,
    SYNTHESIS_MODEL,
    EMBEDDING_MODEL,
    FALLBACK_MODEL
)

logger = logging.getLogger("ModelChecker")

def fetch_openrouter_free_models() -> List[str]:
    """
    Queries GET https://openrouter.ai/api/v1/models and returns list of IDs with $0 pricing.
    """
    try:
        url = f"{OPENROUTER_BASE_URL.rstrip('/')}/models"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            free_models = []
            for m in data.get("data", []):
                model_id = m.get("id", "")
                pricing = m.get("pricing", {})
                prompt_price = float(pricing.get("prompt", 0) or 0)
                completion_price = float(pricing.get("completion", 0) or 0)
                if prompt_price == 0 and completion_price == 0:
                    free_models.append(model_id)
            return free_models
    except Exception as e:
        logger.warning(f"Could not verify OpenRouter free models live: {e}")
    return []

def validate_configured_models() -> Dict[str, bool]:
    """
    Validates configured .env models against live OpenRouter free roster.
    Returns dict of model_env -> is_valid.
    """
    configured = {
        "ORCHESTRATOR_MODEL": ORCHESTRATOR_MODEL,
        "RETRIEVAL_MODEL": RETRIEVAL_MODEL,
        "CRITIC_MODEL": CRITIC_MODEL,
        "SYNTHESIS_MODEL": SYNTHESIS_MODEL,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "FALLBACK_MODEL": FALLBACK_MODEL,
    }
    
    live_free_models = fetch_openrouter_free_models()
    results = {}
    
    if not live_free_models:
        print("[MODEL CHECK] Could not connect to OpenRouter /models endpoint. Assuming configured models are valid.")
        return {k: True for k in configured}

    print("=" * 60)
    print("OpenRouter Free Model Roster Validation")
    print("=" * 60)
    for env_key, model_id in configured.items():
        is_free = model_id in live_free_models or model_id == "openrouter/free" or model_id.endswith(":free")
        results[env_key] = is_free
        status = "VALID ($0 free)" if is_free else "WARNING (Not in $0 roster!)"
        print(f"  {env_key:<20}: {model_id:<40} -> {status}")
    print("-" * 60)
    return results

if __name__ == "__main__":
    validate_configured_models()
