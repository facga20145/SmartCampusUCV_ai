import logging
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger("ErrorHandler")

class ErrorHandler:
    """Manejador centralizado de errores."""
    
    @staticmethod
    def handle_exceptions(func: Callable) -> Callable:
        """Decorador para manejar excepciones en funciones síncronas."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error en {func.__name__}: {e}", exc_info=True)
                raise
        return wrapper
    
    @staticmethod
    def handle_async_exceptions(func: Callable) -> Callable:
        """Decorador para manejar excepciones en funciones asíncronas."""
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error en {func.__name__}: {e}", exc_info=True)
                raise
        return wrapper

