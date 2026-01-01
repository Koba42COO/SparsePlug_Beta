"""
Prime-Sparse SDK Exceptions
===========================

Custom exceptions for the SDK.
"""


class PrimeSparseError(Exception):
    """Base exception for Prime-Sparse SDK."""
    pass


class AuthenticationError(PrimeSparseError):
    """Raised when authentication fails."""
    pass


class RateLimitError(PrimeSparseError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


class ValidationError(PrimeSparseError):
    """Raised when validation fails."""
    pass


class OptimizationError(PrimeSparseError):
    """Raised when optimization fails."""
    pass


class UploadError(PrimeSparseError):
    """Raised when model upload fails."""
    pass


class DownloadError(PrimeSparseError):
    """Raised when model download fails."""
    pass


class QuotaExceededError(PrimeSparseError):
    """Raised when usage quota is exceeded."""
    pass
