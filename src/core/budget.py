"""
Request Budget Guard
Enforces hard daily API request limits across OpenRouter free-tier models to prevent quota exhaustion.
"""
import os
import json
import datetime
import logging
from typing import Dict, Any

from src.config import DAILY_REQUEST_BUDGET, BUDGET_TRACKER_PATH

logger = logging.getLogger("BudgetGuard")
logger.setLevel(logging.INFO)

class BudgetExceededException(Exception):
    """Custom exception raised when the daily OpenRouter request ceiling is reached."""
    pass

class BudgetGuard:
    def __init__(self, tracker_path: str = BUDGET_TRACKER_PATH, max_daily_budget: int = DAILY_REQUEST_BUDGET):
        self.tracker_path = tracker_path
        self.max_daily_budget = max_daily_budget
        os.makedirs(os.path.dirname(self.tracker_path), exist_ok=True)

    def _get_today_key(self) -> str:
        """Returns current date key in YYYY-MM-DD format (UTC)."""
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    def _load_data(self) -> Dict[str, Any]:
        """Loads tracking data from disk."""
        if not os.path.exists(self.tracker_path):
            return {"date": self._get_today_key(), "requests_used": 0}
        
        try:
            with open(self.tracker_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Reset if date key changed
            if data.get("date") != self._get_today_key():
                return {"date": self._get_today_key(), "requests_used": 0}
            return data
        except Exception:
            return {"date": self._get_today_key(), "requests_used": 0}

    def _save_data(self, data: Dict[str, Any]) -> None:
        """Saves tracking data to disk."""
        with open(self.tracker_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_usage(self) -> int:
        """Returns current number of requests made today."""
        data = self._load_data()
        return data.get("requests_used", 0)

    def check_and_increment(self) -> int:
        """
        Checks if request can be made under daily budget limit.
        If under limit, increments count and returns updated count.
        If limit is reached, raises BudgetExceededException.
        """
        data = self._load_data()
        current_used = data.get("requests_used", 0)
        
        if current_used >= self.max_daily_budget:
            msg = f"[BUDGET EXCEEDED] Daily limit of {self.max_daily_budget} requests reached ({current_used}/50 total pool). Halting requests to preserve headroom."
            logger.error(msg)
            raise BudgetExceededException(msg)

        data["requests_used"] = current_used + 1
        self._save_data(data)
        
        print(f"[BUDGET] {data['requests_used']}/50 requests used today (ceiling: {self.max_daily_budget})")
        return data["requests_used"]

# Singleton instance
budget_guard = BudgetGuard()
