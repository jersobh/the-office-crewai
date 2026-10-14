"""Which LLM provider a desk runs on, and how CrewAI reaches it.

A model is written ``<provider>/<model>``:

    gemini/gemini-3.5-flash-lite
    deepseek/deepseek-flash
    fireworks/accounts/fireworks/models/deepseek-v4p1-flash
    openai/<model>      any OpenAI-compatible server at OFFICE_OPENAI_BASE_URL

A bare name, or a Gemini resource name such as models/gemini-2.5-flash, means
Gemini, so older configs keep working. API keys come from the environment and
never appear in logs or errors.

``OFFICE_ALLOWED_PROVIDERS`` turns on anonymous mode: only the listed
providers may be called, Gemini-only extras switch off, and CrewAI telemetry
is disabled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

DEFAULT_TEXT_MODEL = "gemini-3.5-flash-lite"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# provider -> (API key variable, base URL variable, default base URL)
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "gemini": ("GEMINI_API_KEY", "", ""),
    # CrewAI's DeepSeek provider reads DEEPSEEK_BASE_URL itself.
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", ""),
    "fireworks": ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", FIREWORKS_BASE_URL),
    "openai": ("OFFICE_OPENAI_API_KEY", "OFFICE_OPENAI_BASE_URL", ""),
}
_ALIASES = {"google": "gemini"}
# Gemini's own resource names contain a slash; they still mean Gemini.
_GEMINI_RESOURCES = {"models", "tunedmodels"}
_PLACEHOLDER_KEYS = {"MY_GEMINI_API_KEY"}

# Set in anonymous mode before CrewAI is imported, so it never phones home.
_PRIVACY_ENV = {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_TRACING_ENABLED": "false",
}


@dataclass(frozen=True)
class ResolvedModel:
    provider: str
    model: str
    key_env: str
    base_url: str

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


def default_text_model() -> str:
    """The office-wide desk model. ``GEMINI_TEXT_MODEL`` is its older name."""
    return (
        os.getenv("OFFICE_TEXT_MODEL", "").strip()
        or os.getenv("GEMINI_TEXT_MODEL", "").strip()
        or DEFAULT_TEXT_MODEL
    )


def resolve(name: Optional[str] = None) -> ResolvedModel:
    """Parse ``<provider>/<model>``. Empty means the office default."""
    raw = (name or "").strip() or default_text_model()
    prefix, sep, rest = raw.partition("/")
    if not sep or prefix.strip().lower() in _GEMINI_RESOURCES:
        return _resolved("gemini", raw)
    provider = _ALIASES.get(prefix.strip().lower(), prefix.strip().lower())
    if provider not in PROVIDERS:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"Unknown provider '{prefix}' in model '{raw}'. Use one of: {known}.")
    model = rest.strip()
    if not model:
        raise ValueError(f"Model '{raw}' names a provider but no model.")
    return _resolved(provider, model)


def _resolved(provider: str, model: str) -> ResolvedModel:
    key_env, base_env, default_base = PROVIDERS[provider]
    base = (os.getenv(base_env, "").strip() if base_env else "") or default_base
    return ResolvedModel(provider=provider, model=model, key_env=key_env, base_url=base.rstrip("/"))


def api_key(provider: str) -> str:
    key = os.getenv(PROVIDERS[provider][0], "").strip()
    return "" if key in _PLACEHOLDER_KEYS else key


def allowed_providers() -> Optional[set[str]]:
    """The anonymous-mode allow-list, or None when any provider may be called."""
    raw = os.getenv("OFFICE_ALLOWED_PROVIDERS", "").strip()
    if not raw:
        return None
    names = (part.strip().lower() for part in raw.split(","))
    return {_ALIASES.get(name, name) for name in names if name}


def is_allowed(provider: str) -> bool:
    allowed = allowed_providers()
    return allowed is None or provider in allowed


def require(resolved: ResolvedModel) -> str:
    """The API key for a model the office may call. Errors name the variable, never its value."""
    if not is_allowed(resolved.provider):
        raise RuntimeError(
            f"{resolved.label} is blocked: OFFICE_ALLOWED_PROVIDERS does not include '{resolved.provider}'."
        )
    key = api_key(resolved.provider)
    if not key:
        raise RuntimeError(f"{resolved.key_env} is required for {resolved.label}.")
    if resolved.provider == "openai" and not resolved.base_url:
        raise RuntimeError(f"OFFICE_OPENAI_BASE_URL is required for {resolved.label}.")
    return key


def gemini_extras_available() -> bool:
    """Cover images and live voice exist only on Gemini."""
    return is_allowed("gemini") and bool(api_key("gemini"))


def apply_privacy_defaults() -> None:
    """In anonymous mode, turn off CrewAI telemetry and traces. Call before importing crewai."""
    if allowed_providers() is None:
        return
    for name, value in _PRIVACY_ENV.items():
        os.environ[name] = value


def blocked(entries: Iterable[tuple[str, Optional[str]]]) -> list[str]:
    """One line per (setting, provider) pair that OFFICE_ALLOWED_PROVIDERS rules out."""
    allowed = allowed_providers()
    if allowed is None:
        return []
    return [f"{where} uses {provider}" for where, provider in entries if provider and provider not in allowed]


def crew_llm(resolved: ResolvedModel, temperature: float = 0.4) -> Any:
    """The CrewAI LLM for one desk."""
    key = require(resolved)
    apply_privacy_defaults()
    from crewai import LLM

    if resolved.provider == "gemini":
        return LLM(model=f"gemini/{resolved.model}", api_key=key, temperature=temperature)
    if resolved.provider == "deepseek":
        # DeepSeek thinks by default. With tools, every later request must send
        # reasoning_content back or the API answers 400, and CrewAI does not
        # send it. CrewAI passes unknown LLM kwargs into the request body.
        return LLM(
            model=f"deepseek/{resolved.model}",
            api_key=key,
            temperature=temperature,
            extra_body={"thinking": {"type": "disabled"}},
        )
    # Fireworks and any other OpenAI-compatible server. custom_openai pins
    # Chat Completions: CrewAI never switches to the Responses API, which can
    # keep the conversation on the provider (30 days on Fireworks).
    return LLM(
        model=f"openai/{resolved.model}",
        api_key=key,
        base_url=resolved.base_url,
        custom_openai=True,
        api="completions",
        temperature=temperature,
    )
