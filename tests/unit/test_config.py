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
    assert settings.llm_model == "gemini-3.6-flash"
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
