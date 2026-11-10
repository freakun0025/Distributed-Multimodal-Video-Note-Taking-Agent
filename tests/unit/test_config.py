import re
from pathlib import Path

import pytest

from common.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_defaults_match_the_documented_values(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    assert settings.retry_delays == [10, 30, 90]
    assert settings.retry_max_attempts == 5
    assert settings.max_video_seconds == 1200
    assert [str(spec) for spec in settings.llm_chain] == [
        "gemini-3.6-flash:low:90",
        "gemini-3.8-flash:low:120",
        "gemini-3.5-flash-lite:medium:60",
    ]
    assert settings.transcript_langs == ["en", "en-US", "en-GB", "hi"]


def test_comma_separated_lists_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, RETRY_DELAYS="2, 4,8", TRANSCRIPT_LANGS="de,fr")
    assert settings.retry_delays == [2, 4, 8]
    assert settings.transcript_langs == ["de", "fr"]


@pytest.mark.parametrize("value", ["", "0,10", "-1"])
def test_retry_delays_must_be_positive(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    if value == "":
        # Empty means unset -> default, not an error.
        assert _settings(monkeypatch, RETRY_DELAYS=value).retry_delays == [10, 30, 90]
        return
    with pytest.raises(ValueError):
        _settings(monkeypatch, RETRY_DELAYS=value)


def test_the_llm_chain_is_parsed_in_order_with_each_models_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = " gemini-3.5-flash-lite:high:45 , vendor/model:free:low:30.5"
    settings = _settings(monkeypatch, LLM_CHAIN=chain)
    first, second = settings.llm_chain
    assert (first.model, first.thinking, first.timeout_s) == ("gemini-3.5-flash-lite", "high", 45)
    # Split from the right, so model ids with a colon (OpenRouter's ":free") still parse.
    assert (second.model, second.thinking, second.timeout_s) == ("vendor/model:free", "low", 30.5)


@pytest.mark.parametrize(
    "chain",
    [
        "gemini-3.6-flash",  # no thinking level or timeout
        "gemini-3.6-flash:fast:90",  # not a thinking level
        "gemini-3.6-flash:low:0",  # timeout must be positive
        "gemini-3.6-flash:low:90,gemini-3.6-flash:medium:60",  # the same model twice
        ",",  # no model at all
    ],
)
def test_a_bad_llm_chain_is_refused(monkeypatch: pytest.MonkeyPatch, chain: str) -> None:
    with pytest.raises(ValueError):
        _settings(monkeypatch, LLM_CHAIN=chain)


@pytest.mark.parametrize("name", ["LLM_MODEL", "LLM_FALLBACK_MODEL", "LLM_FALLBACK_FROM_ATTEMPT"])
def test_settings_replaced_by_the_chain_are_refused_not_ignored(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    with pytest.raises(ValueError, match=f"{name} -> LLM_"):
        _settings(monkeypatch, **{name: "gemini-3.8-flash"})


def test_attempts_past_the_delay_list_reuse_the_last_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    assert [settings.retry_delay_for(n) for n in range(1, 6)] == [10, 30, 90, 90, 90]


def test_chunk_size_defaults_per_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _settings(monkeypatch).effective_llm_chunk_chars == 60_000
    assert _settings(monkeypatch, LLM_PROVIDER="ollama").effective_llm_chunk_chars == 12_000
    assert (
        _settings(
            monkeypatch, LLM_PROVIDER="ollama", LLM_CHUNK_CHARS="5000"
        ).effective_llm_chunk_chars
        == 5000
    )


def test_secrets_are_not_shown_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, LLM_API_KEY="sk-secret-value")
    assert "sk-secret-value" not in repr(settings)
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "sk-secret-value"


def test_env_example_lists_every_setting() -> None:
    documented = set(
        re.findall(r"^([A-Z][A-Z0-9_]*)=", (REPO_ROOT / ".env.example").read_text(), re.MULTILINE)
    )
    expected = {name.upper() for name in Settings.model_fields}
    assert documented == expected
