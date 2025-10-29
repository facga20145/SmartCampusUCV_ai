import logging
from typing import Optional
from ai.nlp.nlp_core import NLPModule

logger = logging.getLogger("APIUtils")

_nlp_module: Optional[NLPModule] = None

def initialize_nlp_module(config: dict) -> None:
    """Inicializa el módulo NLP globalmente."""
    global _nlp_module
    try:
        _nlp_module = NLPModule(config)
        logger.info("Módulo NLP inicializado correctamente.")
    except Exception as e:
        logger.error(f"Error al inicializar módulo NLP: {e}")
        _nlp_module = None

def get_nlp_module() -> Optional[NLPModule]:
    """Obtiene la instancia del módulo NLP."""
    return _nlp_module

def get_module_status() -> dict:
    """Obtiene el estado de los módulos."""
    global _nlp_module
    status = {
        "status": "online" if _nlp_module and _nlp_module.is_online() else "offline",
        "nlp": "online" if _nlp_module and _nlp_module.is_online() else "offline",
        "model": None
    }
    
    if _nlp_module and hasattr(_nlp_module, "_config"):
        model_config = _nlp_module._config.get("model", {})
        status["model"] = model_config.get("name")
    
    return status

