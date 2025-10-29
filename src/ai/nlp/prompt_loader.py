import os
import logging
from typing import Optional

logger = logging.getLogger("PromptLoader")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML no está instalado. Se usará el prompt de Python como fallback.")

YAML_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.yaml")

def load_system_prompt_template() -> str:
    """
    Carga el template del system prompt desde YAML si está disponible,
    o desde el módulo Python como fallback.
    """
    yaml_template = _load_from_yaml()
    if yaml_template:
        logger.info("System prompt cargado desde YAML.")
        return yaml_template

    logger.info("Usando fallback: system prompt desde módulo Python.")
    # Fallback simple si no hay YAML
    return """Eres un asistente de SmartCampus UCV que ayuda a recomendar actividades sostenibles."""

def _load_from_yaml() -> Optional[str]:
    """Intenta cargar y ensamblar el template desde el archivo YAML modular."""
    if not YAML_AVAILABLE:
        logger.warning("PyYAML no disponible, no se puede leer YAML.")
        return None
    if not os.path.exists(YAML_PATH):
        logger.warning(f"Archivo YAML no encontrado: {YAML_PATH}")
        return None

    try:
        with open(YAML_PATH, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        sections = yaml_data.get("sections")
        if not sections or not isinstance(sections, dict):
            logger.error("El YAML no contiene una clave 'sections' válida.")
            return None

        ordered_keys = [
            "identity",
            "objectives",
            "context",
            "formats",
            "examples"
        ]

        combined_prompt = "\n\n".join(
            [sections[k] for k in ordered_keys if k in sections]
        )

        footer = yaml_data.get("footer")
        if footer:
            combined_prompt += f"\n\n{footer}"

        return combined_prompt.strip()

    except Exception as e:
        logger.error(f"Error al cargar o parsear system_prompt.yaml: {e}")
        return None

