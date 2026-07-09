"""LLM access with a swappable provider — Aura's language brain.

Aura reasons + answers with the best available model, and never dies on stage if the network
does. Two providers behind one interface:

  • OpenRouter (cloud)   — the primary "smart brain": one key, OpenAI-compatible, top models
                           (Llama-3.3-70B, DeepSeek, etc.). Set via env or aura_config.json.
  • Ollama (on-device)   — the always-there fallback (qwen2.5:7b). Used when no cloud key is
                           set, or automatically if a cloud call fails mid-demo.

Embeddings for the RAG copilot always use local nomic-embed-text (cheap, private, offline).
Only the Python stdlib is used for HTTP so the demo adds zero dependencies.

Config precedence (highest first):
  1. Environment: AURA_LLM_PROVIDER, OPENROUTER_API_KEY, OPENROUTER_MODEL
  2. aura-core/aura_config.json  (gitignored):
       { "provider": "openrouter", "openrouter_api_key": "sk-or-...",
         "openrouter_model": "meta-llama/llama-3.3-70b-instruct" }
  3. Defaults below.
"""
from __future__ import annotations

import json
import logging
import math
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

log = logging.getLogger("aura-core")

# ── Ollama (local fallback) ──────────────────────────────────────────
OLLAMA_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen2.5:7b"          # local fallback chat model
EMBED_MODEL = "nomic-embed-text"    # always-local embeddings

# ── Cloud providers (OpenAI-compatible) ──────────────────────────────
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OR_MODEL = "meta-llama/llama-3.3-70b-instruct"

# NVIDIA NIM — free hosted endpoints, generous limits, OpenAI-compatible.
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_NIM_MODEL = "meta/llama-3.3-70b-instruct"

# Gemini — Google's multimodal flash models via the OpenAI-compatible endpoint. Fast + personable;
# the same key powers the real-time Gemini Live voice (see aoede.py). This is the "personalized
# feel" brain for now; on-device Ollama remains the edge fallback + the future direction.
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Providers that speak the OpenAI Chat Completions format (routed through _cloud_endpoint).
_CLOUD = ("openrouter", "nim", "gemini")

_TIMEOUT = 60
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "aura_config.json")
_config_cache: Optional[Dict[str, Any]] = None


def _config() -> Dict[str, Any]:
    """Merge file config + env (env wins). Cached; call reload_config() after editing."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    cfg: Dict[str, Any] = {}
    try:
        if os.path.exists(_CONFIG_PATH):
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
    except Exception as e:
        log.warning("llm: could not read aura_config.json (%s)", e)
    # env overrides
    if os.getenv("OPENROUTER_API_KEY"):
        cfg["openrouter_api_key"] = os.getenv("OPENROUTER_API_KEY")
    if os.getenv("OPENROUTER_MODEL"):
        cfg["openrouter_model"] = os.getenv("OPENROUTER_MODEL")
    if os.getenv("AURA_LLM_PROVIDER"):
        cfg["provider"] = os.getenv("AURA_LLM_PROVIDER")
    _config_cache = cfg
    return cfg


def reload_config() -> Dict[str, Any]:
    """Drop the cache so a freshly-pasted key / model takes effect without a restart."""
    global _config_cache
    _config_cache = None
    return _config()


def _or_key() -> str:
    return (_config().get("openrouter_api_key") or "").strip()


def _or_model() -> str:
    return _config().get("openrouter_model") or DEFAULT_OR_MODEL


def _nim_key() -> str:
    return (_config().get("nim_api_key") or "").strip()


def _nim_model() -> str:
    return _config().get("nim_model") or DEFAULT_NIM_MODEL


def _gemini_key() -> str:
    return (_config().get("gemini_api_key") or "").strip()


def _gemini_model() -> str:
    return _config().get("gemini_model") or DEFAULT_GEMINI_MODEL


def active_provider() -> str:
    """Which chat provider is in effect. An explicit `provider` in config wins (if its key is
    present); otherwise pick the first cloud provider with a key; else local Ollama."""
    forced = (_config().get("provider") or "").strip().lower()
    if forced == "ollama":
        return "ollama"
    if forced == "gemini" and _gemini_key():
        return "gemini"
    if forced == "nim" and _nim_key():
        return "nim"
    if forced == "openrouter" and _or_key():
        return "openrouter"
    # auto: prefer Gemini (multimodal, fast, personable), then other cloud keys, else local edge.
    if _gemini_key():
        return "gemini"
    if _nim_key():
        return "nim"
    if _or_key():
        return "openrouter"
    return "ollama"


def _cloud_endpoint(provider: str):
    """Return (url, headers, model) for an OpenAI-compatible cloud provider."""
    if provider == "nim":
        return NIM_URL, {"Authorization": f"Bearer {_nim_key()}"}, _nim_model()
    if provider == "gemini":
        return GEMINI_URL, {"Authorization": f"Bearer {_gemini_key()}"}, _gemini_model()
    return OPENROUTER_URL, _or_headers(), _or_model()


def _cloud_extra(provider: str) -> Dict[str, Any]:
    """Extra request-body params per provider. Gemini 2.5 flash is a THINKING model — its internal
    reasoning would consume the max_tokens budget and truncate the visible reply, so we turn
    thinking off for these short, conversational turns."""
    if provider == "gemini":
        return {"reasoning_effort": "none"}
    return {}


# ── HTTP helpers ─────────────────────────────────────────────────────
def _post(url: str, body: Dict[str, Any], headers: Dict[str, str], timeout: int = _TIMEOUT) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _or_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_or_key()}",
        "HTTP-Referer": "http://localhost:5173",
        "X-Title": "Aura Driver Co-Pilot",
    }


# ── status ───────────────────────────────────────────────────────────
def ollama_up() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def available() -> bool:
    """Is *some* chat brain reachable? (a cloud key present, or local Ollama up)."""
    return bool(_or_key() or _nim_key() or _gemini_key()) or ollama_up()


def _active_model(prov: Optional[str] = None) -> str:
    prov = prov or active_provider()
    if prov == "nim":
        return _nim_model()
    if prov == "gemini":
        return _gemini_model()
    if prov == "openrouter":
        return _or_model()
    return CHAT_MODEL


def status() -> Dict[str, Any]:
    prov = active_provider()
    return {
        "provider": prov,
        "model": _active_model(prov),
        "cloudKey": bool(_or_key() or _nim_key() or _gemini_key()),
        "ollama": ollama_up(),
        "embedModel": EMBED_MODEL,
    }


# ── chat (routed) ────────────────────────────────────────────────────
def chat(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.4,
    num_predict: int = 220,
) -> str:
    """One-shot chat via the active provider, auto-falling back to Ollama on any cloud error."""
    prov = active_provider()
    if prov in _CLOUD:
        url, headers, cloud_model = _cloud_endpoint(prov)
        try:
            out = _post(
                url,
                {
                    "model": model or cloud_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": num_predict,
                    **_cloud_extra(prov),
                },
                headers,
            )
            return (out["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            log.warning("%s chat failed (%s) — falling back to Ollama", prov, e)
    # Ollama
    try:
        out = _post(
            f"{OLLAMA_URL}/api/chat",
            {
                "model": model or CHAT_MODEL,
                "messages": messages,
                "stream": False,
                "keep_alive": "30m",  # keep the model resident so no cold-start mid-demo
                "options": {"temperature": temperature, "num_predict": num_predict},
            },
            {},
        )
        return (out.get("message", {}) or {}).get("content", "").strip()
    except Exception as e:
        log.warning("ollama chat failed (%s) — is a brain available?", e)
        return ""


def chat_stream(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.4,
    num_predict: int = 220,
) -> Iterator[str]:
    """Stream tokens from the active provider (cloud SSE or Ollama JSON lines).
    Falls back to a one-shot Ollama call if a cloud stream can't be opened."""
    prov = active_provider()
    if prov in _CLOUD:
        url, headers, cloud_model = _cloud_endpoint(prov)
        try:
            yield from _stream_cloud(url, headers, messages, model or cloud_model, temperature, num_predict, _cloud_extra(prov))
            return
        except Exception as e:
            log.warning("%s stream failed (%s) — falling back to Ollama", prov, e)
    try:
        yield from _stream_ollama(messages, model or CHAT_MODEL, temperature, num_predict)
    except Exception as e:
        log.warning("ollama stream failed (%s)", e)
        return


def _stream_cloud(url, headers, messages, model, temperature, num_predict, extra=None) -> Iterator[str]:
    """Stream from any OpenAI-compatible endpoint (OpenRouter / NIM / Gemini) via SSE."""
    body = {
        "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": num_predict, "stream": True, **(extra or {}),
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                piece = chunk["choices"][0].get("delta", {}).get("content", "")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if piece:
                yield piece


def _stream_ollama(messages, model, temperature, num_predict) -> Iterator[str]:
    body = {
        "model": model, "messages": messages, "stream": True, "keep_alive": "30m",
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        for raw in resp:
            line = raw.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            piece = (chunk.get("message", {}) or {}).get("content", "")
            if piece:
                yield piece
            if chunk.get("done"):
                break


# ── embeddings (always local) ────────────────────────────────────────
def embed(text: str, *, model: str = EMBED_MODEL) -> Optional[List[float]]:
    """Embed one string with local nomic-embed-text. Returns None on failure."""
    try:
        out = _post(f"{OLLAMA_URL}/api/embeddings", {"model": model, "prompt": text, "keep_alive": "30m"}, {}, timeout=30)
        vec = out.get("embedding")
        return vec if isinstance(vec, list) and vec else None
    except Exception as e:
        log.warning("llm.embed failed (%s)", e)
        return None


def cosine(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity — no numpy dependency for a handful of chunks."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
