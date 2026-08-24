from myai.health import healthcheck


def test_healthcheck_is_ok() -> None:
    result = healthcheck()
    assert result["status"] == "ok"
    assert result["app"] == "MY-AI"
    assert result["version"] == "v5"


def test_healthcheck_shape_is_stable() -> None:
    result = healthcheck()
    assert set(result) == {"status", "app", "version"}
