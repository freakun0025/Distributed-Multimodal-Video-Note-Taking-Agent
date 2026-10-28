import asyncio

import httpx
import pytest

from common.config import Settings
from gateway.main import create_app
from gateway.routes.health import HealthCheck


async def _ok() -> None:
    return None


async def _down() -> None:
    raise ConnectionRefusedError


async def _hang() -> None:
    await asyncio.sleep(10)


async def _get_healthz(checks: dict[str, HealthCheck]) -> httpx.Response:
    app = create_app(Settings(_env_file=None, healthcheck_timeout_s=0.05))  # type: ignore[call-arg]
    app.state.health_checks = checks
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/healthz")


async def test_healthz_is_200_when_every_dependency_is_reachable() -> None:
    response = await _get_healthz({"postgres": _ok, "rabbitmq": _ok, "redis": _ok})
    assert response.status_code == 200
    assert response.json() == {
        "gateway": "ok",
        "postgres": "ok",
        "rabbitmq": "ok",
        "redis": "ok",
    }


@pytest.mark.parametrize("broken", ["postgres", "rabbitmq", "redis"])
async def test_healthz_is_503_and_names_the_failing_dependency(broken: str) -> None:
    checks: dict[str, HealthCheck] = {"postgres": _ok, "rabbitmq": _ok, "redis": _ok}
    checks[broken] = _down
    response = await _get_healthz(checks)
    assert response.status_code == 503
    assert response.json()[broken] == "down"
    assert [name for name, value in response.json().items() if value == "down"] == [broken]


async def test_a_hanging_dependency_times_out_as_down() -> None:
    response = await _get_healthz({"postgres": _ok, "rabbitmq": _hang, "redis": _ok})
    assert response.status_code == 503
    assert response.json()["rabbitmq"] == "down"
