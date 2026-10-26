"""
Unified OpenRouter API Client
Provides budget-aware, cache-accelerated, and rotation-resilient OpenAI-compatible calls to OpenRouter.
Includes offline mock mode when OPENROUTER_API_KEY is set to placeholder values.
"""
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    FALLBACK_MODEL
)
from src.core.budget import budget_guard, BudgetExceededException
from src.core.cache import response_cache

logger = logging.getLogger("OpenRouterClient")

def _is_placeholder_key(key: str) -> bool:
    """Checks if API key is missing or placeholder."""
    if not key or "your_openrouter_api_key" in key.lower() or key == "placeholder_key":
        return True
    return False

def _generate_mock_embedding(text: str, dim: int = 1536) -> List[float]:
    """
    Generates a term-frequency aware normalized vector for offline similarity testing.
    Keywords in the text produce deterministic high-magnitude components at specific indices.
    """
    text_lower = text.lower()
    words = re_words = [w.strip(".,()[]:;-") for w in text_lower.split() if len(w) > 2]
    
    vec = [0.0] * dim
    
    # Base hash component
    base_hash = int(hashlib.md5(text_lower.encode('utf-8')).hexdigest(), 16)
    for i in range(dim):
        vec[i] = ((base_hash >> (i % 32)) & 1) * 0.05

    # Word-feature component: hash each word to a vector index
    for word in words:
        word_hash = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
        idx = word_hash % dim
        vec[idx] += 1.0

    # L2 normalize vector
    norm = (sum(x * x for x in vec)) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec

class OpenRouterClient:
    def __init__(self, api_key: str = OPENROUTER_API_KEY, base_url: str = OPENROUTER_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key or "placeholder_key")

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1000,
        bypass_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Executes a chat completion call with caching, budgeting, retry backoff, and fallback handling.
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # 1. Check local response cache first
        if not bypass_cache:
            cached = response_cache.get(model, payload)
            if cached:
                return cached

        # 2. Check & increment daily request budget guard
        budget_guard.check_and_increment()

        # 3. Check for placeholder API key
        if _is_placeholder_key(self.api_key):
            print(f"[OFFLINE MOCK CHAT] OPENROUTER_API_KEY is placeholder. Generating mock response for model '{model}'")
            mock_content = f"Mock synthesis response for offline testing. Query parsed successfully."
            result = {
                "model": model,
                "content": mock_content,
                "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
                "latency_ms": 10
            }
            response_cache.set(model, payload, result)
            return result

        # 4. Call live OpenRouter API with backoff and 1-time model fallback
        target_model = model
        start_time = time.time()
        
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                latency_ms = int((time.time() - start_time) * 1000)
                
                content = response.choices[0].message.content or ""
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0
                total_tokens = response.usage.total_tokens if response.usage else 0

                result = {
                    "model": target_model,
                    "content": content,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    },
                    "latency_ms": latency_ms
                }

                print(f"[API CALL] Model: '{target_model}' | Latency: {latency_ms}ms | Tokens: {total_tokens}")
                response_cache.set(model, payload, result)
                return result

            except Exception as e:
                err_str = str(e)
                if attempt == 0 and ("404" in err_str or "410" in err_str or "not found" in err_str.lower()):
                    print(f"[MODEL FALLBACK] Model '{target_model}' returned 404/410. Falling back to '{FALLBACK_MODEL}'")
                    target_model = FALLBACK_MODEL
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"OpenRouter call failed for model '{target_model}': {e}")
                    raise e

        raise RuntimeError("Chat completion failed after fallback attempts.")

    def embed_texts(
        self,
        model: str,
        texts: List[str],
        batch_size: int = 16
    ) -> List[List[float]]:
        """
        Generates vector embeddings for a list of texts in small batches via OpenRouter /embeddings endpoint.
        Uses term-frequency aware mock embeddings when OPENROUTER_API_KEY is placeholder.
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"texts": batch}

            # Check cache for batch
            cached = response_cache.get(model, payload)
            if cached and "embeddings" in cached:
                all_embeddings.extend(cached["embeddings"])
                continue

            # Check budget
            budget_guard.check_and_increment()

            # Handle offline mock mode when API key is placeholder
            if _is_placeholder_key(self.api_key):
                print(f"[OFFLINE MOCK EMBEDDING] Generating {len(batch)} mock vectors for model '{model}'")
                batch_embeddings = [_generate_mock_embedding(t) for t in batch]
                all_embeddings.extend(batch_embeddings)
                response_cache.set(model, payload, {"embeddings": batch_embeddings, "latency_ms": 5})
                continue

            start_time = time.time()
            try:
                response = self.client.embeddings.create(
                    model=model,
                    input=batch
                )
                latency_ms = int((time.time() - start_time) * 1000)
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

                total_tokens = response.usage.total_tokens if response.usage else 0
                print(f"[EMBEDDING CALL] Batch size: {len(batch)} | Latency: {latency_ms}ms | Tokens: {total_tokens}")

                response_cache.set(model, payload, {"embeddings": batch_embeddings, "latency_ms": latency_ms})
            except Exception as e:
                logger.error(f"Embedding call failed for model '{model}': {e}")
                raise e

        return all_embeddings

# Singleton instance
openrouter_client = OpenRouterClient()
