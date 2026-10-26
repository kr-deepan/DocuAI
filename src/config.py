"""
Configuration loader for OpenRouter API, model roles, request budgeting, and vector store paths.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent

# OpenRouter API settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Free model tiering configuration
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "poolside/laguna-xs-2.1:free")
RETRIEVAL_MODEL = os.getenv("RETRIEVAL_MODEL", "poolside/laguna-xs-2.1:free")
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "poolside/laguna-s-2.1:free")
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "openrouter/free")

# Request budgeting & rate limiting
DAILY_REQUEST_BUDGET = int(os.getenv("DAILY_REQUEST_BUDGET", "45"))
BUDGET_TRACKER_PATH = os.path.join(BASE_DIR, "data", "budget_tracker.json")
RESPONSE_CACHE_PATH = os.path.join(BASE_DIR, "data", "response_cache.db")

# Vector store settings
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(BASE_DIR, "data", "chroma_db"))
