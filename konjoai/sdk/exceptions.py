"""Exception hierarchy raised by the Kyro Python SDK client."""

from __future__ import annotations


class KyroError(Exception):
    """Base exception for all Kyro SDK errors."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class KyroAuthError(KyroError):
    """Raised on 401 / 403 responses."""


class KyroRateLimitError(KyroError):
    """Raised on 429 responses.

    ``retry_after`` is populated from the ``Retry-After`` response header when
    present.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class KyroTimeoutError(KyroError):
    """Raised when the underlying HTTP request times out."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=None)


class KyroNotFoundError(KyroError):
    """Raised on 404 responses."""
