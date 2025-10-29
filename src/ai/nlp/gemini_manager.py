import logging
import os
from typing import Dict, Any, Optional
import google.generativeai as genai

logger = logging.getLogger("GeminiManager")

class GeminiManager:
    """Gestiona la conectividad con Google Gemini API."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Inicializa GeminiManager.

        Args:
            model_config: Configuración del modelo (nombre, temperatura, max_tokens)
        """
        self._model_config: Dict[str, Any] = model_config
        # Usar gemini-1.5-flash que tiene mejor tier gratuito
        self._model_name: Optional[str] = model_config.get("name", "gemini-1.5-flash")
        self._online: bool = False
        
        # Obtener API key desde variable de entorno
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logger.warning("GEMINI_API_KEY no está configurada. Gemini no estará disponible.")
            self._online = False
            return
        
        try:
            # Configurar Gemini
            genai.configure(api_key=api_key)
            self._online = True
            logger.info(f"GeminiManager inicializado con modelo: {self._model_name}")
        except Exception as e:
            logger.error(f"Error al inicializar Gemini: {e}")
            self._online = False

    def is_online(self) -> bool:
        """Indica si el módulo Gemini está en línea y listo para usarse."""
        return self._online

    def get_model(self):
        """Obtiene el modelo de Gemini configurado."""
        if not self._online:
            return None
        
        try:
            generation_config = {
                "temperature": self._model_config.get("temperature", 0.7),
                "max_output_tokens": self._model_config.get("max_tokens", 1024),
            }
            
            model = genai.GenerativeModel(
                model_name=self._model_name,
                generation_config=generation_config
            )
            return model
        except Exception as e:
            logger.error(f"Error al obtener modelo Gemini: {e}")
            return None

    def reload(self, model_config: Dict[str, Any]):
        """Recarga la configuración del modelo."""
        logger.info("Recargando configuración de Gemini...")
        self._model_config = model_config
        self._model_name = model_config.get("name", "gemini-pro")

