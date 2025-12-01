import logging
import os
from typing import Dict, Any, Optional
from groq import Groq, AsyncGroq

logger = logging.getLogger("GroqManager")

class GroqManager:
    """Gestiona la conectividad con Groq API."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Inicializa GroqManager.

        Args:
            model_config: Configuración del modelo (nombre, temperatura, max_tokens)
        """
        self._model_config: Dict[str, Any] = model_config
        # Usar llama3-8b-8192 como default si no se especifica
        self._model_name: str = model_config.get("name", "llama3-8b-8192")
        self._online: bool = False
        self._client: Optional[AsyncGroq] = None
        
        # Obtener API key desde variable de entorno
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            logger.warning("GROQ_API_KEY no está configurada. Groq no estará disponible.")
            self._online = False
            return
        
        try:
            # Configurar Cliente Groq Asíncrono para uso principal
            self._client = AsyncGroq(api_key=api_key)
            
            # Verificar conexión usando un cliente síncrono temporal (solo para inicialización)
            try:
                sync_client = Groq(api_key=api_key)
                sync_client.models.list()
                self._online = True
                logger.info(f"GroqManager inicializado con modelo: {self._model_name}")
            except Exception as e:
                logger.error(f"Error al conectar con Groq API (verificación): {e}")
                self._online = False
                
        except Exception as e:
            logger.error(f"Error al inicializar cliente Groq: {e}")
            self._online = False

    def is_online(self) -> bool:
        """Indica si el módulo Groq está en línea y listo para usarse."""
        return self._online

    async def generate_content(self, prompt: str) -> Optional[str]:
        """
        Genera contenido usando el modelo configurado de forma asíncrona.
        
        Args:
            prompt: El prompt para enviar al modelo.
            
        Returns:
            El texto generado o None si hubo error.
        """
        if not self._online or not self._client:
            logger.error("Groq no está disponible")
            return None
        
        try:
            # Llamada asíncrona a Groq
            completion = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self._model_config.get("temperature", 0.7),
                max_tokens=self._model_config.get("max_tokens", 1024),
                top_p=1,
                stream=False,
                stop=None,
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error durante la generación de contenido en Groq: {e}")
            return None

    def reload(self, model_config: Dict[str, Any]):
        """Recarga la configuración del modelo."""
        logger.info("Recargando configuración de Groq...")
        self._model_config = model_config
        self._model_name = model_config.get("name", "llama3-8b-8192")
