"""OpenAI-compatible chat-completion client with retries and usage tracking.

The optional live runner reads endpoint settings from environment variables.
All experiment code must call LLMs through this client so that every call is
logged and retry behaviour is consistent.
"""
import sys
import threading
import time
from typing import Dict, List, Optional

import requests

import os

# 支持系统代理
PROXIES = {
    'http': os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY'),
    'https': os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY'),
}
# 移除None值
PROXIES = {k: v for k, v in PROXIES.items() if v}

from .config import (
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    load_api_config,
)


class LLMClient:
    def __init__(self, model: Optional[str] = None, timeout: int = 240):
        # Load config based on model
        cfg = load_api_config(model)
        self.base_url = cfg["base_url"]
        self.api_key = cfg["api_key"]
        self.model = model or cfg["model"]
        self.timeout = timeout
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": 0}

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_err: Optional[Exception] = None
        for attempt in range(LLM_MAX_RETRIES):
            try:
                resp = requests.post(url, headers=headers, json=payload, proxies=PROXIES or None, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                with self._lock:
                    self.stats["calls"] += 1
                    self.stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    self.stats["completion_tokens"] += usage.get("completion_tokens", 0)
                if content:
                    return content
                last_err = RuntimeError("empty completion content")
            except (requests.exceptions.RequestException, KeyError, ValueError) as exc:
                last_err = exc
                with self._lock:
                    self.stats["errors"] += 1
                if "429" in str(exc) or "Too Many Requests" in str(exc):
                    # Rate limit - longer backoff
                    wait = min(5 + 3 ** attempt, 60)
                else:
                    wait = min(3 + 2 ** attempt, 30)
                if attempt < LLM_MAX_RETRIES - 1:
                    sys.stderr.write(f"[LLM retry {attempt + 1}/{LLM_MAX_RETRIES}] {type(exc).__name__}: {str(exc)[:100]}, waiting {wait}s\n")
                    sys.stderr.flush()
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {LLM_MAX_RETRIES} retries: {last_err}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kw) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kw)
