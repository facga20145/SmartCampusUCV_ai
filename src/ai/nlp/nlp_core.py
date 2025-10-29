import asyncio
import logging
import re
import json
import os
from typing import Optional, Dict, Any, List
from datetime import datetime

from ai.nlp.gemini_manager import GeminiManager
from ai.nlp.prompt_creator import create_system_prompt, create_recommendation_prompt

logger = logging.getLogger("NLPModule")

RECOMMENDATION_JSON_REGEX = re.compile(
    r"(?:GENERAR_RECOMENDACION_JSON|Generar_recomendacion_JSON):\s*({.*?})",
    re.DOTALL | re.IGNORECASE
)

class NLPModule:
    """Clase principal para el procesamiento NLP con integración a Gemini para recomendaciones."""

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el módulo NLP.
        
        Args:
            config: Configuración completa de la aplicación
        """
        self._config = config
        model_config = config.get("model", {
            "name": "gemini-1.5-flash",
            "temperature": 0.7,
            "max_tokens": 1024
        })
        self._gemini_manager = GeminiManager(model_config)
        self._online = self._gemini_manager.is_online()
        logger.info("NLPModule inicializado.")

    def is_online(self) -> bool:
        """Devuelve True si el módulo NLP está online."""
        return self._gemini_manager.is_online()

    async def generate_recommendations(
        self,
        usuario_id: int,
        preferencias: List[Dict[str, Any]],
        actividades_disponibles: List[Dict[str, Any]],
        historial_participacion: List[Dict[str, Any]] = None,
        hobbies: Optional[str] = None,
        intereses: Optional[str] = None,
        user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Genera recomendaciones de actividades usando Gemini API.
        
        Args:
            usuario_id: ID del usuario
            preferencias: Lista de preferencias del usuario
            actividades_disponibles: Lista de actividades disponibles
            historial_participacion: Historial de participación del usuario
            hobbies: Hobbies del usuario
            intereses: Intereses del usuario
            user_query: Consulta opcional del usuario
            
        Returns:
            Diccionario con las recomendaciones generadas
        """
        logger.info(f"Generando recomendaciones para usuario {usuario_id}")

        if not self.is_online():
            return {
                "error": "El módulo NLP está fuera de línea",
                "recomendaciones": []
            }

        if not actividades_disponibles:
            return {
                "error": "No hay actividades disponibles",
                "recomendaciones": []
            }

        historial_participacion = historial_participacion or []
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Crear prompts
        system_prompt = create_system_prompt(
            config=self._config,
            usuario_id=usuario_id,
            preferencias=preferencias,
            actividades_disponibles=actividades_disponibles,
            historial_participacion=historial_participacion,
            hobbies=hobbies,
            intereses=intereses,
            fecha_actual=fecha_actual
        )

        user_prompt = create_recommendation_prompt(
            user_query=user_query or "Recomiéndame actividades basadas en mis preferencias",
            preferencias=preferencias,
            actividades_disponibles=actividades_disponibles,
            historial_participacion=historial_participacion,
            hobbies=hobbies,
            intereses=intereses
        )

        # Generar respuesta con Gemini
        recommendations = []
        model = self._gemini_manager.get_model()

        if not model:
            return {
                "error": "No se pudo obtener el modelo de Gemini",
                "recomendaciones": []
            }

        try:
            # Combinar system prompt y user prompt para Gemini
            # Gemini no tiene separación system/user, así que combinamos
            full_prompt = f"""{system_prompt}

{user_prompt}"""

            # Generar respuesta con Gemini
            response = model.generate_content(full_prompt)
            full_response_content = response.text

            # Extraer recomendaciones del JSON
            recommendations = self._extract_recommendations(full_response_content, actividades_disponibles)

            logger.info(f"Generadas {len(recommendations)} recomendaciones para usuario {usuario_id}")

            return {
                "recomendaciones": recommendations,
                "response_text": full_response_content
            }

        except Exception as e:
            logger.error(f"Error al generar recomendaciones con Gemini: {e}")
            return {
                "error": f"Error al generar recomendaciones: {str(e)}",
                "recomendaciones": []
            }

    def _extract_recommendations(self, response_text: str, actividades_disponibles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extrae recomendaciones del texto de respuesta.
        
        Args:
            response_text: Texto de respuesta del modelo
            actividades_disponibles: Lista de actividades disponibles
            
        Returns:
            Lista de recomendaciones extraídas
        """
        recommendations = []
        actividad_ids_set = {act.get("id") for act in actividades_disponibles if act.get("id")}

        # Buscar todos los JSON de recomendación en el texto
        matches = RECOMMENDATION_JSON_REGEX.findall(response_text)

        for match in matches:
            try:
                recommendation_data = json.loads(match)
                actividad_id = recommendation_data.get("actividad_id")

                # Validar que el ID existe
                if actividad_id and actividad_id in actividad_ids_set:
                    # Encontrar la actividad correspondiente
                    actividad = next(
                        (act for act in actividades_disponibles if act.get("id") == actividad_id),
                        None
                    )

                    if actividad:
                        recommendations.append({
                            "actividad_id": actividad_id,
                            "titulo": actividad.get("titulo", ""),
                            "categoria": actividad.get("categoria", ""),
                            "razon": recommendation_data.get("razon", ""),
                            "puntuacion": recommendation_data.get("puntuacion", 0.0),
                            "actividad": actividad
                        })
                else:
                    logger.warning(f"Actividad ID {actividad_id} no encontrada en actividades disponibles")

            except json.JSONDecodeError as e:
                logger.error(f"Error al decodificar JSON de recomendación: {e}")
            except Exception as e:
                logger.error(f"Error al procesar recomendación: {e}")

        # Ordenar por puntuación descendente
        recommendations.sort(key=lambda x: x.get("puntuacion", 0.0), reverse=True)

        return recommendations

