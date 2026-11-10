"""LLM provider interface shared by Gemini, Ollama and the fake (ARCHITECTURE §8.2, O-6)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from common.config import Settings
from common.errors import ErrorCode, ExternalApi, PermanentError
from common.limiter import Limiter

if TYPE_CHECKING:
    from worker.llm.health import ModelHealth


@dataclass(frozen=True)
class PromptImage:
    """A JPEG shown to the model at its place in the prompt (a candidate frame, D34)."""

    data: bytes = field(repr=False)
    mime_type: str = "image/jpeg"


PromptPart = str | PromptImage


@dataclass(frozen=True)
class LlmRequest:
    system: str  # rules for the model
    parts: tuple[PromptPart, ...]  # the transcript part and its context, images in place
    schema: Mapping[str, Any]  # JSON Schema the answer must follow (structured output)

    @property
    def prompt(self) -> str:
        """The text alone (logs, tests, text-only providers)."""
        return "".join(part for part in self.parts if isinstance(part, str))

    @property
    def images(self) -> tuple[PromptImage, ...]:
        return tuple(part for part in self.parts if isinstance(part, PromptImage))


class LlmProvider(Protocol):
    async def generate_json(self, request: LlmRequest) -> str:
        """The model's answer as JSON text; acquires the `llm` limiter token itself."""
        ...

    @property
    def answered_by(self) -> str | None:
        """The model that gave the last answer, if the provider knows (logs only)."""
        ...


def make_provider(
    settings: Settings, limiter: Limiter, health: ModelHealth, attempt: int = 1
) -> LlmProvider:
    """The configured provider; with FAKE_APIS=true, Gemini's client talks to the fakes service.

    The models of LLM_CHAIN in order, skipping those `health` says any worker has benched. From
    attempt LLM_STEP_DOWN_FROM_ATTEMPT on, any 5xx/timeout steps down to the next model at
    once, so a long provider outage still ends in notes rather than the DLQ (D31, D36).
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
        chain=settings.llm_chain,
        health=health,
        image_resolution=settings.llm_image_resolution,
        step_down_on_overload=attempt >= settings.llm_step_down_from_attempt,
        base_url=base_url,
    )
