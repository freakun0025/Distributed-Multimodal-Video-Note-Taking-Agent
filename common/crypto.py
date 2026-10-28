"""Fernet encryption of the Notion token in broker messages (ARCHITECTURE §6.2; invariant 2)."""

from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken
from pydantic import SecretStr

from common.errors import ErrorCode, PermanentError


class TokenCipher:
    """Encrypts/decrypts the Notion token. Gateway and workers share TOKEN_ENCRYPTION_KEY."""

    def __init__(self, key: SecretStr | str) -> None:
        raw = key.get_secret_value() if isinstance(key, SecretStr) else key
        try:
            self._fernet = Fernet(raw.encode())
        except ValueError as exc:
            # Never include the key in the message.
            raise ValueError("TOKEN_ENCRYPTION_KEY is not a valid Fernet key") from exc

    def encrypt(self, token: str) -> str:
        return self._fernet.encrypt(token.encode()).decode()

    def decrypt(self, token_enc: str) -> str:
        try:
            return self._fernet.decrypt(token_enc.encode()).decode()
        except InvalidToken as exc:
            # Wrong key or tampered message: retrying cannot fix it.
            raise PermanentError(
                ErrorCode.TOKEN_DECRYPT_FAILED,
                "Notion token could not be decrypted (TOKEN_ENCRYPTION_KEY mismatch?)",
            ) from exc


def generate_key() -> str:
    """New TOKEN_ENCRYPTION_KEY (used by `make fernet-key`)."""
    return Fernet.generate_key().decode()


if __name__ == "__main__":
    import sys

    sys.stdout.write(generate_key() + "\n")
