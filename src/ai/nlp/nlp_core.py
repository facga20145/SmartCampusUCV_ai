import asyncio
import logging
import re
import json
import os
from typing import Optional, Dict, Any, List
from datetime import datetime

from ai.nlp.groq_manager import GroqManager
from ai.nlp.prompt_creator import create_system_prompt, create_recommendation_prompt

logger = logging.getLogger("NLPModule")

# Regex mejorado para capturar JSONs incluso si tienen saltos de línea
RECOMMENDATION_JSON_REGEX = re.compile(
    r"(?:GENERAR_RECOMENDACION_JSON|Generar_recomendacion_JSON|GENERAR_RECOMENDACION|generar_recomendacion):\s*({[^}]*})",
    re.DOTALL | re.IGNORECASE
)

# Regex alternativo para JSONs en múltiples líneas
RECOMMENDATION_JSON_MULTILINE_REGEX = re.compile(
    r"(?:GENERAR_RECOMENDACION_JSON|Generar_recomendacion_JSON):\s*({[\s\S]*?})",
    re.IGNORECASE
)

# Regex para buscar IDs de actividades en el texto
ACTIVITY_ID_REGEX = re.compile(r'"actividad_id"\s*:\s*(\d+)', re.IGNORECASE)

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
            "name": "llama3-8b-8192",
            "temperature": 0.7,
            "max_tokens": 1024
        })
        self._groq_manager = GroqManager(model_config)
        self._online = self._groq_manager.is_online()
        logger.info("NLPModule inicializado.")

    def is_online(self) -> bool:
        """Devuelve True si el módulo NLP está online."""
        return self._groq_manager.is_online()

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

        # Generar respuesta con Groq
        try:
            # Combinar system prompt y user prompt
            full_prompt = f"""{system_prompt}

{user_prompt}"""

            # Generar respuesta
            full_response_content = self._groq_manager.generate_content(full_prompt)
            
            if not full_response_content:
                 return {
                    "error": "Error al generar respuesta con Groq (respuesta vacía)",
                    "recomendaciones": []
                }

            # Log la respuesta completa para debugging (primeros 500 caracteres)
            logger.debug(f"Respuesta de Groq (primeros 500 chars): {full_response_content[:500]}")
            logger.info(f"Respuesta completa de Groq: {full_response_content}")

            # Extraer recomendaciones del JSON
            recommendations = self._extract_recommendations(full_response_content, actividades_disponibles)

            logger.info(f"Generadas {len(recommendations)} recomendaciones para usuario {usuario_id}")
            if len(recommendations) == 0:
                logger.warning(f"No se pudieron extraer recomendaciones. Respuesta fue: {full_response_content[:200]}")

            return {
                "recomendaciones": recommendations,
                "response_text": full_response_content
            }

        except Exception as e:
            logger.error(f"Error al generar recomendaciones con Groq: {e}")
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

        processed_ids = set()  # Para evitar duplicados

        # Buscar todos los JSON de recomendación en el texto (intentar múltiples formatos)
        matches = RECOMMENDATION_JSON_REGEX.findall(response_text)
        if not matches:
            # Intentar con regex multilinea
            matches = RECOMMENDATION_JSON_MULTILINE_REGEX.findall(response_text)
        
        # Si aún no hay matches, buscar IDs directamente en el texto o por nombre
        if not matches:
            found_ids = ACTIVITY_ID_REGEX.findall(response_text)
            logger.info(f"No se encontraron JSONs completos, pero se encontraron IDs: {found_ids}")
            for actividad_id_str in found_ids:
                try:
                    actividad_id = int(actividad_id_str)
                    if actividad_id in actividad_ids_set and actividad_id not in processed_ids:
                        actividad = next(
                            (act for act in actividades_disponibles if act.get("id") == actividad_id),
                            None
                        )
                        if actividad:
                            recommendations.append({
                                "actividad_id": actividad_id,
                                "titulo": actividad.get("titulo", ""),
                                "categoria": actividad.get("categoria", ""),
                                "razon": "Recomendada basándose en tus preferencias",
                                "puntuacion": 0.8,
                                "actividad": actividad
                            })
                            processed_ids.add(actividad_id)
                except ValueError:
                    continue
            
            # Si tampoco hay IDs, intentar buscar por título de actividad mencionado en el texto
            if not found_ids and not recommendations:
                logger.info("Intentando extraer recomendaciones por nombre de actividad...")
                for actividad in actividades_disponibles:
                    titulo = actividad.get("titulo", "").strip()
                    if titulo and len(titulo) > 3 and titulo in response_text:
                        # Verificar que no esté ya procesada
                        actividad_id = actividad.get("id")
                        if actividad_id and actividad_id not in processed_ids:
                            recommendations.append({
                                "actividad_id": actividad_id,
                                "titulo": titulo,
                                "categoria": actividad.get("categoria", ""),
                                "razon": "Recomendada basándose en tus preferencias y mencionada en la respuesta",
                                "puntuacion": 0.85,
                                "actividad": actividad
                            })
                            processed_ids.add(actividad_id)
                            logger.info(f"✅ Recomendación encontrada por nombre: {titulo} (ID: {actividad_id})")
                            break  # Solo la primera mencionada

        # Procesar JSONs encontrados
        for match in matches:
            try:
                # Limpiar el match (eliminar saltos de línea y espacios extra)
                cleaned_match = match.replace('\n', '').replace('\r', '').strip()
                # Intentar parsear como JSON
                recommendation_data = json.loads(cleaned_match)
                actividad_id = recommendation_data.get("actividad_id")

                # Validar que el ID existe y no ha sido procesado
                if actividad_id and actividad_id in actividad_ids_set and actividad_id not in processed_ids:
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
                            "razon": recommendation_data.get("razon", "Recomendada basándose en tus preferencias"),
                            "puntuacion": recommendation_data.get("puntuacion", 0.8),
                            "actividad": actividad
                        })
                        processed_ids.add(actividad_id)
                        logger.info(f"Recomendación extraída: actividad_id={actividad_id}, título={actividad.get('titulo', '')}")
                else:
                    if actividad_id:
                        logger.warning(f"Actividad ID {actividad_id} no encontrada o ya procesada")

            except json.JSONDecodeError as e:
                logger.warning(f"Error al decodificar JSON de recomendación: {e}. JSON: {match[:100]}")
                # Intentar extraer solo el ID si el JSON está mal formateado
                id_match = ACTIVITY_ID_REGEX.search(match)
                if id_match:
                    try:
                        actividad_id = int(id_match.group(1))
                        if actividad_id in actividad_ids_set and actividad_id not in processed_ids:
                            actividad = next(
                                (act for act in actividades_disponibles if act.get("id") == actividad_id),
                                None
                            )
                            if actividad:
                                recommendations.append({
                                    "actividad_id": actividad_id,
                                    "titulo": actividad.get("titulo", ""),
                                    "categoria": actividad.get("categoria", ""),
                                    "razon": "Recomendada basándose en tus preferencias",
                                    "puntuacion": 0.8,
                                    "actividad": actividad
                                })
                                processed_ids.add(actividad_id)
                    except (ValueError, AttributeError):
                        pass
            except Exception as e:
                logger.error(f"Error al procesar recomendación: {e}")

        # Ordenar por puntuación descendente
        recommendations.sort(key=lambda x: x.get("puntuacion", 0.0), reverse=True)
        
        logger.info(f"Total de recomendaciones extraídas: {len(recommendations)}")

        return recommendations

