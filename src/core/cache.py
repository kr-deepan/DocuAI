"""
Local Response Cache
SQLite-backed caching for OpenRouter chat completions and embedding calls.
Prevents redundant API calls and conserves daily request quota during dev & ingestion iterations.
"""
import os
import json
import sqlite3
import hashlib
from typing import Optional, Any, Dict

from src.config import RESPONSE_CACHE_PATH

class ResponseCache:
    def __init__(self, db_path: str = RESPONSE_CACHE_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initializes SQLite cache schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    response_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def generate_key(self, model: str, payload: Any) -> str:
        """Generates deterministic SHA-256 hash key for a request payload."""
        serialized = json.dumps({"model": model, "payload": payload}, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, model: str, payload: Any) -> Optional[Dict[str, Any]]:
        """Retrieves cached response dict if present, else None."""
        key = self.generate_key(model, payload)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT response_json FROM response_cache WHERE cache_key = ?", (key,))
            row = cursor.fetchone()
            if row:
                print(f"[CACHE HIT] Serving cached response for model '{model}' (0 quota used)")
                return json.loads(row[0])
        return None

    def set(self, model: str, payload: Any, response_data: Dict[str, Any]) -> None:
        """Saves response data in cache."""
        key = self.generate_key(model, payload)
        response_json = json.dumps(response_data)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO response_cache (cache_key, response_json) VALUES (?, ?)",
                (key, response_json)
            )
            conn.commit()

# Singleton instance
response_cache = ResponseCache()
