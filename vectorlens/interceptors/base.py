"""Abstract base class for library interceptors."""

from abc import ABC, abstractmethod


class BaseInterceptor(ABC):
    """Base class for all monkey-patch interceptors."""

    @abstractmethod
    def install(self) -> None:
        """Install the interceptor by monkey-patching the target library."""
        pass

    @abstractmethod
    def uninstall(self) -> None:
        """Restore original functions."""
        pass

    @abstractmethod
    def is_installed(self) -> bool:
        """Return True if interceptor is currently installed."""
        pass
