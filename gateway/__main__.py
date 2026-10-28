"""`python -m gateway`: JSON logging, then uvicorn on port 8000 (ARCHITECTURE §10)."""

from __future__ import annotations

import uvicorn

from common.config import get_settings
from common.logging import configure_logging

PORT = 8000  # container port; the host-facing URL is GATEWAY_BASE_URL


def main() -> None:
    configure_logging(get_settings().log_level, service="gateway")
    # log_config=None keeps uvicorn on our JSON root handler instead of its own formatter.
    uvicorn.run("gateway.main:app", host="0.0.0.0", port=PORT, log_config=None)


if __name__ == "__main__":
    main()
