from lectra.core.errors import (
    ConfigurationError,
    LectraRAGError,
)


def test_custom_error_inheritence() -> None:
    assert issubclass(ConfigurationError, LectraRAGError)
