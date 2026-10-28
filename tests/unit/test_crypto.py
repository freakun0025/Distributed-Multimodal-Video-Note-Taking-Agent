import pytest
from pydantic import SecretStr

from common.crypto import TokenCipher, generate_key
from common.errors import ErrorCode, PermanentError

TOKEN = "ntn_example_token_value"


def test_token_round_trips_through_encryption() -> None:
    cipher = TokenCipher(SecretStr(generate_key()))
    encrypted = cipher.encrypt(TOKEN)
    assert TOKEN not in encrypted
    assert cipher.decrypt(encrypted) == TOKEN


def test_gateway_and_worker_with_the_same_key_interoperate() -> None:
    key = generate_key()
    assert TokenCipher(key).decrypt(TokenCipher(key).encrypt(TOKEN)) == TOKEN


def test_wrong_key_is_a_permanent_decrypt_failure() -> None:
    encrypted = TokenCipher(generate_key()).encrypt(TOKEN)
    with pytest.raises(PermanentError) as caught:
        TokenCipher(generate_key()).decrypt(encrypted)
    assert caught.value.code is ErrorCode.TOKEN_DECRYPT_FAILED


def test_invalid_key_error_does_not_echo_the_key() -> None:
    with pytest.raises(ValueError) as caught:
        TokenCipher("definitely-not-a-fernet-key")
    assert "definitely-not-a-fernet-key" not in str(caught.value)
