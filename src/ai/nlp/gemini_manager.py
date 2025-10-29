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
                model_names = [m.name for m in available_models]
                logger.info(f"Modelos disponibles: {len(model_names)} modelos encontrados")
                
                # Filtrar modelos que soportan generateContent (heurísticas basadas en nombre)
                # Excluir: embedding, imagen, veo, gemma, aqa, learnlm, audio, live, thinking, tts, image-generation
                valid_models = []
                excluded_keywords = ['embedding', 'imagen', 'veo', 'gemma', 'aqa', 'learnlm', 
                                   'computer-use', 'robotics', 'audio', 'live', 'thinking', 
                                   'tts', 'image-generation']
                
                for model_name in model_names:
                    if not model_name.startswith('models/gemini-'):
                        continue
                    # Excluir modelos especializados
                    if any(x in model_name.lower() for x in excluded_keywords):
                        continue
                    valid_models.append(model_name)
                
                logger.info(f"Modelos válidos para generateContent: {len(valid_models)} encontrados")
                
                # Prioridad de selección:
                # 1. modelos con "-latest" y "flash" (ej: models/gemini-flash-latest)
                # 2. modelos con "flash" estable (ej: models/gemini-2.5-flash)
                # 3. modelos con "-latest" y "pro" (ej: models/gemini-pro-latest)
                # 4. cualquier otro modelo flash
                # 5. cualquier otro modelo pro
                
                selected_model = None
                
                # Prioridad 1: flash-latest
                for model in valid_models:
                    if 'flash' in model.lower() and 'latest' in model.lower():
                        selected_model = model
                        logger.info(f"Modelo seleccionado (flash-latest): {model}")
                        break
                
                # Prioridad 2: flash estable (sin preview, sin latest)
                if not selected_model:
                    for model in valid_models:
                        if 'flash' in model.lower() and 'preview' not in model.lower() and 'latest' not in model.lower():
                            selected_model = model
                            logger.info(f"Modelo seleccionado (flash estable): {model}")
                            break
                
                # Prioridad 3: pro-latest
                if not selected_model:
                    for model in valid_models:
                        if 'pro' in model.lower() and 'latest' in model.lower():
                            selected_model = model
                            logger.info(f"Modelo seleccionado (pro-latest): {model}")
                            break
                
                # Prioridad 4: cualquier modelo flash
                if not selected_model:
                    for model in valid_models:
                        if 'flash' in model.lower():
                            selected_model = model
                            logger.info(f"Modelo seleccionado (flash): {model}")
                            break
                
                # Prioridad 5: cualquier modelo pro
                if not selected_model:
                    for model in valid_models:
                        if 'pro' in model.lower():
                            selected_model = model
                            logger.info(f"Modelo seleccionado (pro): {model}")
                            break
                
                # Prioridad 6: cualquier modelo gemini válido
                if not selected_model and valid_models:
                    selected_model = valid_models[0]
                    logger.info(f"Modelo seleccionado (primer disponible): {selected_model}")
                
                if selected_model:
                    # Actualizar el nombre del modelo a usar
                    self._available_model_name = selected_model
                    self._model_name = selected_model
                    logger.info(f"✅ Usando modelo: {self._model_name}")
                else:
                    logger.warning("⚠️ No se encontró ningún modelo compatible con generateContent. Usando modelo configurado.")
                    
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

