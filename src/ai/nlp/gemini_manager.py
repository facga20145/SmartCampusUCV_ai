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
        # Usar gemini-pro que está disponible en el tier gratuito
        self._model_name: Optional[str] = model_config.get("name", "gemini-pro")
        self._online: bool = False
        self._available_model_name: Optional[str] = None
        
        # Obtener API key desde variable de entorno
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logger.warning("GEMINI_API_KEY no está configurada. Gemini no estará disponible.")
            self._online = False
            return
        
        try:
            # Configurar Gemini
            genai.configure(api_key=api_key)
            
            # Listar modelos disponibles y encontrar uno que soporte generateContent
            try:
                available_models = genai.list_models()
                logger.info(f"Modelos disponibles: {[m.name for m in available_models]}")
                
                # Buscar un modelo que soporte generateContent
                for model_info in available_models:
                    # Verificar si el modelo soporta generateContent
                    if hasattr(model_info, 'supported_generation_methods'):
                        if 'generateContent' in model_info.supported_generation_methods:
                            # Preferir modelos que contengan "flash" o "pro" en el nombre
                            model_name = model_info.name
                            if 'flash' in model_name.lower():
                                self._available_model_name = model_name
                                logger.info(f"Modelo encontrado (preferido flash): {model_name}")
                                break
                            elif 'pro' in model_name.lower() and not self._available_model_name:
                                self._available_model_name = model_name
                                logger.info(f"Modelo encontrado (preferido pro): {model_name}")
                            elif not self._available_model_name:
                                self._available_model_name = model_name
                                logger.info(f"Modelo encontrado: {model_name}")
                    else:
                        # Si no tiene el atributo, intentar con el nombre directamente
                        if not self._available_model_name:
                            self._available_model_name = model_info.name
                            logger.info(f"Modelo encontrado (sin verificación de métodos): {model_info.name}")
                
                if self._available_model_name:
                    # Actualizar el nombre del modelo a usar
                    self._model_name = self._available_model_name
                    logger.info(f"Usando modelo: {self._model_name}")
                else:
                    logger.warning("No se encontró ningún modelo compatible con generateContent")
                    
            except Exception as e:
                logger.warning(f"No se pudieron listar modelos disponibles: {e}. Usando modelo configurado: {self._model_name}")
            
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

