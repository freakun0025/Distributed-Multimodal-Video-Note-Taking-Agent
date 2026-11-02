"""LLM provider interface shared by Gemini, Ollama and the fake (ARCHITECTURE §8.2, O-6)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from common.config import Settings
from common.errors import ErrorCode, ExternalApi, PermanentError
from common.limiter import Limiter


@dataclass(frozen=True)
class LlmRequest:
    system: str  # rules for the model
    prompt: str  # the transcript part and its context
    schema: Mapping[str, Any]  # JSON Schema the answer must follow (structured output)


class LlmProvider(Protocol):
    async def generate_json(self, request: LlmRequest) -> str:
        """The model's answer as JSON text; acquires the `llm` limiter token itself."""
        ...


def make_provider(settings: Settings, limiter: Limiter, attempt: int = 1) -> LlmProvider:
    """The configured provider; with FAKE_APIS=true, Gemini's client talks to the fakes service.

    From attempt LLM_FALLBACK_FROM_ATTEMPT on, an overloaded primary model hands over to the
    fallback model, so a long provider outage still ends in notes rather than the DLQ (D31).
    """
    if settings.llm_provider != "gemini":
        # Ollama arrives in M4 (O-6); FAKE_APIS=true already replaces "fake" (ARCHITECTURE §14).
        raise NotImplementedError(f"LLM_PROVIDER={settings.llm_provider} is not built yet")
    from worker.llm.gemini import GeminiProvider  # google-genai loads only when it is used

    if settings.fake_apis:
        api_key, base_url = "fake-key", f"{settings.fakes_url}/llm"
    elif settings.llm_api_key is None:
        raise PermanentError(
            ErrorCode.UPSTREAM_REJECTED, "LLM_API_KEY is not set", api=ExternalApi.LLM
        )
    else:
        api_key, base_url = settings.llm_api_key.get_secret_value(), None
    return GeminiProvider(
        limiter,
        api_key=api_key,
        model=settings.llm_model,
        fallback_model=settings.llm_fallback_model,
        thinking_level=settings.llm_thinking_level,
        timeout_s=settings.llm_timeout_s,
        fallback_on_overload=attempt >= settings.llm_fallback_from_attempt,
        base_url=base_url,
    )
