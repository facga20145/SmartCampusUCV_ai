import json
from typing import Dict, Any, List
from ai.nlp.prompt_loader import load_system_prompt_template

def create_system_prompt(
    config: Dict[str, Any],
    usuario_id: int,
    preferencias: List[Dict[str, Any]],
    actividades_disponibles: List[Dict[str, Any]],
    historial_participacion: List[Dict[str, Any]],
    hobbies: Any = None,
    intereses: Any = None,
    fecha_actual: str = ""
) -> str:
    """
    Crea un prompt del sistema personalizado para recomendaciones de actividades.
    
    Args:
        config: Configuración de la aplicación
        usuario_id: ID del usuario
        preferencias: Lista de preferencias del usuario
        actividades_disponibles: Lista de actividades disponibles
        historial_participacion: Historial de actividades ya realizadas
        fecha_actual: Fecha actual en formato legible
        
    Returns:
        String con el prompt del sistema completo
    """
    template = load_system_prompt_template()
    
    # Formatear preferencias
    preferencias_str = json.dumps(preferencias, ensure_ascii=False, indent=2) if preferencias else "Ninguna preferencia registrada"
    
    # Formatear actividades disponibles (limitar para no sobrecargar el contexto)
    actividades_limitadas = actividades_disponibles[:30]  # Limitar a 30 actividades
    actividades_str = json.dumps(actividades_limitadas, ensure_ascii=False, indent=2)
    
    # Formatear historial
    historial_str = json.dumps(historial_participacion, ensure_ascii=False, indent=2) if historial_participacion else "Sin historial previo"
    
    # Formatear hobbies e intereses
    hobbies_str = hobbies if hobbies else "No especificados"
    intereses_str = intereses if intereses else "No especificados"
    
    # Reemplazar variables en el template
    prompt = template.format(
        assistant_name=config.get("assistant_name", "SmartCampus Assistant"),
        language=config.get("language", "es"),
        usuario_id=usuario_id,
        preferencias=preferencias_str,
        actividades_disponibles=actividades_str,
        historial_participacion=historial_str,
        hobbies=hobbies_str,
        intereses=intereses_str,
        fecha_actual=fecha_actual
    )
    
    return prompt

def create_recommendation_prompt(
    user_query: str,
    preferencias: List[Dict[str, Any]],
    actividades_disponibles: List[Dict[str, Any]],
    historial_participacion: List[Dict[str, Any]],
    hobbies: Any = None,
    intereses: Any = None
) -> str:
    """
    Crea un prompt para generar recomendaciones específicas.
    
    Args:
        user_query: Consulta del usuario (opcional)
        preferencias: Preferencias del usuario
        actividades_disponibles: Actividades disponibles
        historial_participacion: Historial de participación
        
    Returns:
        String con el prompt de recomendación
    """
    hobbies_info = f"Hobbies del usuario: {hobbies}" if hobbies else "Hobbies: No especificados"
    intereses_info = f"Intereses del usuario: {intereses}" if intereses else "Intereses: No especificados"
    
    prompt = f"""Basándote en las preferencias, hobbies, intereses del usuario y las actividades disponibles, genera recomendaciones personalizadas.

Preferencias del usuario: {json.dumps(preferencias, ensure_ascii=False)}
{hobbies_info}
{intereses_info}

ACTIVIDADES DISPONIBLES (SCHEMA):
Cada actividad tiene esta estructura:
- "id": número entero (ÚSALO en actividad_id)
- "categoria": string
- "titulo": string
- "descripcion": string o null
- "fecha": string (YYYY-MM-DD)
- "lugar": string
- "nivel_sostenibilidad": número (0-5) o null

Actividades disponibles: {json.dumps(actividades_disponibles[:30], ensure_ascii=False)}

Historial de participación: {json.dumps(historial_participacion, ensure_ascii=False)}

REGLAS CRÍTICAS:
1. Usa SOLO los IDs que existen en la lista de actividades disponibles
2. El campo "id" de cada actividad es el que debes usar en "actividad_id" del JSON
3. Prioriza actividades que coincidan con hobbies e intereses del usuario
4. Busca palabras clave de hobbies/intereses en: titulo, descripcion, categoria

"""
    
    if user_query:
        prompt += f"Consulta del usuario: {user_query}\n\n"
    
    prompt += """FORMATO DE RESPUESTA REQUERIDO:
1. Para cada recomendación (máximo 5), incluye:
   - Texto descriptivo explicando la actividad
   - Línea separadora "---"
   - GENERAR_RECOMENDACION_JSON: {"actividad_id": [número exacto del campo "id"], "razon": "...", "puntuacion": 0.0-1.0}

2. EJEMPLO CORRECTO:
---
GENERAR_RECOMENDACION_JSON: {"actividad_id": 1, "razon": "Coincide con tus intereses", "puntuacion": 0.9}

3. IMPORTANTE:
   - actividad_id DEBE ser el número exacto del campo "id" de una actividad de la lista
   - NO inventes IDs
   - NO uses ```json```, escribe el JSON directamente en una línea"""
    
    return prompt

