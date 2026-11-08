"""
Swappable LLM layer.

One function -- call_llm() -- behind three providers chosen by env var:

    LLM_PROVIDER=gemini             (default)  GEMINI_API_KEY
    LLM_PROVIDER=openai_compatible             OPENAI_API_KEY + OPENAI_BASE_URL
                                               covers OpenRouter / Groq / vLLM /
                                               anything OpenAI-shaped Kuku provides
    LLM_PROVIDER=mock                          no key, deterministic, offline

Swapping the model is an env var, never a code edit. That is a stated
requirement of the brief ("we may connect stronger models through APIs we
provide").

Designed so the free tier is never the bottleneck:
  * disk cache keyed on sha256(provider|model|system|user) -- re-runs are free,
    so iterating on ranking never re-pays for selection, and the interview
    re-run is instant
  * token-bucket rate limiter (RPM from config)
  * exponential backoff honouring Retry-After on 429/5xx
  * every raw response written to raw/ before parsing, so a crash loses nothing
  * real token counts logged, so cost claims are measured and not invented
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    raise SystemExit("pip install requests")

try:
    from dotenv import dotenv_values

    # Precedence: explicit environment variables (e.g. `LLM_PROVIDER=mock python ...`)
    # beat .env, EXCEPT API keys, where .env wins so a stale key exported in the shell
    # cannot silently shadow the project's key.
    for _k, _v in dotenv_values().items():
        if _v and (_k not in os.environ or _k.endswith("_API_KEY")):
            os.environ[_k] = _v
except Exception:
    pass


GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


class LLMError(Exception):
    pass


class QuotaExhausted(LLMError):
    """A DAILY quota is used up. Retrying cannot help, so we fail fast and remember it."""


# model -> human-readable reason. Once a model's daily quota is gone, every later call to
# it raises instantly (no HTTP, no sleeping) instead of retrying and then splitting
# batches into more doomed calls. Cache hits are unaffected.
EXHAUSTED: Dict[str, str] = {}


def quota_exhausted() -> Dict[str, str]:
    return dict(EXHAUSTED)


# --------------------------------------------------------------------------
# rate limiting
# --------------------------------------------------------------------------

class RateLimiter:
    """Token bucket. Blocks until a request slot is free."""

    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        self.interval = 60.0 / self.rpm
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_at = max(now, self._next_at) + self.interval


# --------------------------------------------------------------------------
# usage accounting
# --------------------------------------------------------------------------

@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    cached_prompt_tokens: int = 0
    cached_output_tokens: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    wall_s: float = 0.0
    by_model: Dict[str, int] = field(default_factory=dict)

    def add(self, model: str, prompt: int, output: int) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.output_tokens += output
        self.by_model[model] = self.by_model.get(model, 0) + 1

    def summary(self) -> Dict[str, Any]:
        return {
            "api_calls": self.calls,
            "cache_hits": self.cached,
            "retries": self.retries,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.prompt_tokens + self.output_tokens,
            # tokens a full run needs regardless of cache -- use THIS for cost claims
            "prompt_tokens_incl_cache": self.prompt_tokens + self.cached_prompt_tokens,
            "output_tokens_incl_cache": self.output_tokens + self.cached_output_tokens,
            "wall_seconds": round(self.wall_s, 1),
            "calls_by_model": self.by_model,
        }


USAGE = Usage()


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "gemini")
    # Defaults checked against Google's model + deprecation pages on 2026-11-08:
    # gemini-2.5-flash is deprecated (new users blocked, shutdown 2026-12-01).
    # Model IDs churn -- override with LLM_MODEL / JUDGE_MODEL, never edit code.
    model: str = os.getenv("LLM_MODEL", "gemini-3.6-flash")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    # Generous headroom: newer Gemini models think by default and thinking tokens
    # can count against this budget (a MAX_TOKENS stop raises a clear error).
    max_output_tokens: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "16384"))
    # Gemini 2.5 models think by default; thinking silently eats the output
    # budget and can return an empty candidate. 0 keeps runs fast and
    # predictable. Raise it if a stronger model is swapped in.
    thinking_budget: int = int(os.getenv("LLM_THINKING_BUDGET", "0"))
    rpm: int = int(os.getenv("LLM_RPM", "10"))
    timeout_s: int = int(os.getenv("LLM_TIMEOUT_S", "180"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "4"))
    cache_dir: str = os.getenv("LLM_CACHE_DIR", ".cache/llm")
    raw_dir: str = os.getenv("LLM_RAW_DIR", "raw")

    def with_model(self, model: Optional[str]) -> "LLMConfig":
        if not model or model == self.model:
            return self
        clone = LLMConfig(**{**self.__dict__})
        clone.model = model
        return clone


_LIMITERS: Dict[int, RateLimiter] = {}
_LIMITER_LOCK = threading.Lock()


def _limiter(rpm: int) -> RateLimiter:
    with _LIMITER_LOCK:
        if rpm not in _LIMITERS:
            _LIMITERS[rpm] = RateLimiter(rpm)
        return _LIMITERS[rpm]


# --------------------------------------------------------------------------
# JSON handling
# --------------------------------------------------------------------------

def extract_json(text: str) -> Any:
    """Parse model output into JSON, tolerating fences and surrounding prose."""
    if text is None:
        raise LLMError("empty model response")
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost balanced {...} or [...] in the text.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        if start == -1:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMError(f"could not parse JSON from response: {text[:300]!r}")


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------

def _post_with_retry(url: str, payload: dict, headers: dict, cfg: LLMConfig) -> dict:
    last_err: Optional[Exception] = None
    for attempt in range(cfg.max_retries):
        _limiter(cfg.rpm).acquire()
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=cfg.timeout_s)
        except requests.RequestException as exc:
            last_err = exc
            USAGE.retries += 1
            time.sleep(min(2**attempt + random.random(), 30))
            continue

        if r.status_code == 200:
            return r.json()

        if r.status_code in (408, 429, 500, 502, 503, 504):
            USAGE.retries += 1
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = 2**attempt
            else:
                delay = 2**attempt + random.random()
            # Gemini puts a retry hint in the error body for quota errors.
            if r.status_code == 429:
                m = re.search(r'"retryDelay"\s*:\s*"(\d+)s"', r.text)
                if m:
                    delay = max(delay, float(m.group(1)))
                # A per-day quota (or a retry hint of minutes) will not clear within
                # this run: stop retrying and flag the model as exhausted.
                if "PerDay" in r.text or "per day" in r.text.lower() or (m and float(m.group(1)) > 120):
                    lim = re.search(r"limit:\s*(\d+)", r.text)
                    reason = (
                        f"daily quota exhausted for model {cfg.model}"
                        + (f" (limit {lim.group(1)}/day)" if lim else "")
                        + (f"; retry in about {int(float(m.group(1)) // 60)} min" if m else "")
                    )
                    raise QuotaExhausted(reason)    # callers decide whether another key exists
            last_err = LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
            time.sleep(min(delay, 60))
            continue

        raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")

    raise LLMError(f"giving up after {cfg.max_retries} attempts: {last_err}")


def _gemini_schema(schema: Any) -> Any:
    """Gemini's Schema wants upper-case type names and rejects some JSON-Schema keys."""
    drop = {"additionalProperties", "$schema", "title", "default"}
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k in drop:
                continue
            if k == "type" and isinstance(v, str):
                out[k] = v.upper()
            else:
                out[k] = _gemini_schema(v)
        return out
    if isinstance(schema, list):
        return [_gemini_schema(x) for x in schema]
    return schema


# (model, key index) pairs whose DAILY quota is used up. Keys are never logged or stored.
KEY_EXHAUSTED: set = set()
KEY_SWITCHES: List[str] = []


def _gemini_keys() -> List[str]:
    """GEMINI_API_KEY, then GEMINI_API_KEY_2, _3 ... (blank values ignored, duplicates removed)."""
    names = ["GEMINI_API_KEY", "GOOGLE_API_KEY"] + [f"GEMINI_API_KEY_{i}" for i in range(2, 6)]
    keys: List[str] = []
    for n in names:
        v = (os.getenv(n) or "").strip()
        if v and v not in keys:
            keys.append(v)
    return keys


def _call_gemini(system: str, user: str, schema: Optional[dict], cfg: LLMConfig) -> str:
    keys = _gemini_keys()
    if not keys:
        raise LLMError(
            "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
            "or run with LLM_PROVIDER=mock."
        )
    gen: Dict[str, Any] = {
        "temperature": cfg.temperature,
        "maxOutputTokens": cfg.max_output_tokens,
        "responseMimeType": "application/json",
    }
    # responseSchema is opt-in: the prompts already spell out the output format
    # and responseMimeType guarantees valid JSON. A rejected schema would 400 the
    # whole run, so the safe path is the default. Enable with
    # LLM_USE_RESPONSE_SCHEMA=1 once verified against your model.
    if schema and os.getenv("LLM_USE_RESPONSE_SCHEMA") == "1":
        gen["responseSchema"] = _gemini_schema(schema)
    # thinkingBudget=0 is valid on 2.5 Flash/Flash-Lite but rejected by 2.5 Pro.
    if "2.5" in cfg.model and "pro" not in cfg.model:
        gen["thinkingConfig"] = {"thinkingBudget": cfg.thinking_budget}

    payload = {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen,
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    # Use the first key whose daily quota for this model is not exhausted; on a daily-quota
    # 429, remember it and fall through to the next key. Only when EVERY key is exhausted
    # does QuotaExhausted propagate.
    data = None
    last_exc: Optional[QuotaExhausted] = None
    for i, key in enumerate(keys):
        if (cfg.model, i) in KEY_EXHAUSTED:
            continue
        try:
            data = _post_with_retry(
                GEMINI_ENDPOINT.format(model=cfg.model) + f"?key={key}",
                payload,
                {"Content-Type": "application/json"},
                cfg,
            )
            break
        except QuotaExhausted as exc:
            KEY_EXHAUSTED.add((cfg.model, i))
            last_exc = exc
            if any((cfg.model, j) not in KEY_EXHAUSTED for j in range(len(keys))):
                KEY_SWITCHES.append(f"{cfg.model}: key {i + 1} -> next key")
    if data is None:
        why = str(last_exc) if last_exc else f"daily quota exhausted for model {cfg.model}"
        raise QuotaExhausted(why + (f" (all {len(keys)} keys)" if len(keys) > 1 else ""))

    usage = data.get("usageMetadata", {})
    USAGE.add(
        cfg.model,
        int(usage.get("promptTokenCount", 0)),
        int(usage.get("candidatesTokenCount", 0)),
    )

    candidates = data.get("candidates") or []
    if not candidates:
        raise LLMError(f"no candidates returned: {json.dumps(data)[:400]}")
    cand = candidates[0]
    if cand.get("finishReason") == "MAX_TOKENS":
        raise LLMError(
            "response hit MAX_TOKENS -- reduce batch size or raise LLM_MAX_OUTPUT_TOKENS"
        )
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    if not text.strip():
        raise LLMError(f"empty candidate text (finishReason={cand.get('finishReason')})")
    return text


def _call_openai_compatible(
    system: str, user: str, schema: Optional[dict], cfg: LLMConfig
) -> str:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("GROQ_API_KEY")
    base = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    if not key:
        raise LLMError("OPENAI_API_KEY (or OPENROUTER_API_KEY / GROQ_API_KEY) is not set.")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_output_tokens,
        "response_format": {"type": "json_object"},
    }
    data = _post_with_retry(
        f"{base}/chat/completions",
        payload,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        cfg,
    )
    usage = data.get("usage", {}) or {}
    USAGE.add(
        cfg.model,
        int(usage.get("prompt_tokens", 0)),
        int(usage.get("completion_tokens", 0)),
    )
    choices = data.get("choices") or []
    if not choices:
        raise LLMError(f"no choices returned: {json.dumps(data)[:400]}")
    return choices[0]["message"]["content"]


def _call_mock(system: str, user: str, schema: Optional[dict], cfg: LLMConfig) -> str:
    """
    Deterministic offline stand-in. Lets the whole pipeline run with no key,
    so the prototype always demos. Imported lazily to keep this module thin.
    """
    from mock_provider import mock_response

    USAGE.add("mock", len(user) // 4, 200)
    return mock_response(system, user, schema)


PROVIDERS = {
    "gemini": _call_gemini,
    "openai_compatible": _call_openai_compatible,
    "mock": _call_mock,
}


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def call_llm(
    system: str,
    user: str,
    schema: Optional[dict] = None,
    *,
    cfg: Optional[LLMConfig] = None,
    model: Optional[str] = None,
    tag: str = "call",
    use_cache: bool = True,
) -> Any:
    """
    Send one prompt, get parsed JSON back.

    `tag` names the raw-response file so the raw/ folder is readable.
    `model` overrides the configured model for this call (used to run the
    Pass-2 judge on a different model from the Pass-1 selector).
    """
    cfg = (cfg or LLMConfig()).with_model(model)
    provider = PROVIDERS.get(cfg.provider)
    if provider is None:
        raise LLMError(
            f"unknown LLM_PROVIDER={cfg.provider!r}; expected one of {sorted(PROVIDERS)}"
        )

    fingerprint = prompt_fingerprint(cfg, system, user, schema)

    cache_path = os.path.join(cfg.cache_dir, f"{fingerprint}.json")
    raw_path = os.path.join(cfg.raw_dir, f"{tag}.{fingerprint}.txt")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            entry = json.loads(fh.read())
        USAGE.cached += 1
        USAGE.cached_prompt_tokens += int(entry.get("prompt_tokens", 0))
        USAGE.cached_output_tokens += int(entry.get("output_tokens", 0))
        # Re-materialise the raw response so a re-run into a fresh out/ folder
        # still ships the unedited model output the brief asks for.
        if entry.get("raw_text") is not None and not os.path.exists(raw_path):
            os.makedirs(cfg.raw_dir, exist_ok=True)
            with open(raw_path, "w", encoding="utf-8") as fh:
                fh.write(entry["raw_text"])
        return entry["parsed"]

    if cfg.model in EXHAUSTED:
        raise QuotaExhausted(EXHAUSTED[cfg.model])

    started = time.monotonic()
    before_p, before_o = USAGE.prompt_tokens, USAGE.output_tokens
    try:
        text = provider(system, user, schema, cfg)
    except QuotaExhausted as exc:
        EXHAUSTED[cfg.model] = str(exc)      # every key is gone: fail all later calls instantly
        raise
    call_p = USAGE.prompt_tokens - before_p
    call_o = USAGE.output_tokens - before_o

    # Persist the raw response BEFORE parsing, so a parse failure is debuggable
    # and a crash never loses an API call we already paid for.
    os.makedirs(cfg.raw_dir, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as fh:
        fh.write(text)

    try:
        parsed = extract_json(text)
    except LLMError:
        # One repair attempt: ask for JSON only, no cache.
        USAGE.retries += 1
        repair = provider(
            system,
            user + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object, no prose, no code fences.",
            schema,
            cfg,
        )
        with open(
            os.path.join(cfg.raw_dir, f"{tag}.{fingerprint}.repair.txt"), "w", encoding="utf-8"
        ) as fh:
            fh.write(repair)
        parsed = extract_json(repair)

    USAGE.wall_s += time.monotonic() - started

    if use_cache:
        os.makedirs(cfg.cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "tag": tag,
                    "model": cfg.model,
                    "parsed": parsed,
                    "raw_text": text,
                    "prompt_tokens": call_p,
                    "output_tokens": call_o,
                },
                fh,
                ensure_ascii=False,
            )

    return parsed


def prompt_fingerprint(cfg: LLMConfig, system: str, user: str, schema: Optional[dict]) -> str:
    """The cache key for a call. Shared with estimate.py so a dry run can tell, exactly,
    which calls are already cached and therefore free."""
    return hashlib.sha256(
        json.dumps(
            [cfg.provider, cfg.model, cfg.temperature, system, user, schema],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:20]


def is_cached(cfg: LLMConfig, system: str, user: str, schema: Optional[dict]) -> bool:
    return os.path.exists(os.path.join(cfg.cache_dir, f"{prompt_fingerprint(cfg, system, user, schema)}.json"))


def usage_summary() -> Dict[str, Any]:
    return USAGE.summary()


if __name__ == "__main__":
    cfg = LLMConfig()
    print(f"provider={cfg.provider} model={cfg.model} rpm={cfg.rpm}")
    out = call_llm(
        "You are a test harness. Reply with JSON.",
        'Return exactly {"ok": true, "note": "<one short word>"}',
        {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}, "note": {"type": "string"}},
            "required": ["ok", "note"],
        },
        tag="smoketest",
    )
    print("response:", out)
    print("usage:", json.dumps(usage_summary(), indent=2))
