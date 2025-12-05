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
    
    # Crear lista de palabras clave para matching
    keywords = []
    if hobbies:
        keywords.extend(hobbies.lower().split(','))
        keywords.extend(hobbies.lower().split())
    if intereses:
        keywords.extend(intereses.lower().split(','))
        keywords.extend(intereses.lower().split())
    keywords = [k.strip() for k in keywords if len(k.strip()) > 2]
    keywords_str = ", ".join(keywords[:20]) if keywords else "ninguna"
    
    prompt = f"""Basándote en las preferencias, hobbies, intereses del usuario y las actividades disponibles, genera recomendaciones personalizadas.

Preferencias del usuario: {json.dumps(preferencias, ensure_ascii=False)}
{hobbies_info}
{intereses_info}

PALABRAS CLAVE DEL USUARIO PARA MATCHING: {keywords_str}
**IMPORTANTE**: Busca estas palabras clave en el título, descripción y categoría de las actividades. Por ejemplo:
- Si el usuario tiene "Deporte" en hobbies → busca actividades con categoría "deportiva" o palabras relacionadas
- Si el usuario tiene "Arte" en intereses → busca actividades con "arte", "artística", "cultural" en categoría/título

ACTIVIDADES DISPONIBLES (SCHEMA):
Cada actividad tiene esta estructura:
- "id": número entero (ÚSALO en actividad_id)
- "categoria": string (ej: "deportiva", "cultural", "ambiental")
- "titulo": string
- "descripcion": string o null
- "fecha": string (YYYY-MM-DD)
- "lugar": string
- "nivel_sostenibilidad": número (0-5) o null

Actividades disponibles: {json.dumps(actividades_disponibles[:30], ensure_ascii=False)}

Historial de participación: {json.dumps(historial_participacion, ensure_ascii=False)}

REGLAS CRÍTICAS:
1. Usa SOLO los IDs que existen en la lista de actividades disponibles (verifica que el ID existe antes de usarlo)
2. El campo "id" de cada actividad es el que debes usar en "actividad_id" del JSON
3. **PRIORIDAD DE MATCHING**: Busca actividades que COINCIDAN con los hobbies e intereses del usuario:
   - Si el usuario tiene "Futbol" o "Deporte" → busca categoría "deportiva"
   - Si el usuario tiene "Arte" o "Danza" → busca categoría "artistica" o "cultural"
   - Si el usuario tiene "Música" → busca actividades de música o categoría "artistica"
4. **REGLA IMPORTANTE - SI HAY COINCIDENCIAS**: Si encuentras actividades que coinciden con los hobbies/intereses:
   - Recomienda SOLAMENTE esas actividades con puntuación alta (0.8-0.95)
   - NO incluyas actividades de otras categorías
   - Ejemplo: Si el usuario le gusta "Futbol" y hay una actividad deportiva, SOLO recomienda esa
5. **SOLO SI NO HAY COINCIDENCIAS**: ÚNICAMENTE si NO hay NINGUNA actividad que coincida:
   - Primero di: "Actualmente no hay actividades de [categoría del usuario] disponibles."
   - Luego agrega: "Sin embargo, te invitamos a participar en estas otras actividades mientras tanto:"
   - Recomienda las alternativas con puntuación baja (0.4-0.5)
6. NUNCA mezcles actividades que coinciden con alternativas - es una cosa o la otra

"""
    
    if user_query:
        prompt += f"Consulta del usuario: {user_query}\n\n"
    
    prompt += """FORMATO DE RESPUESTA OBLIGATORIO (DEBES SEGUIRLO EXACTAMENTE):

INSTRUCCIONES CRÍTICAS:
1. DEBES incluir SIEMPRE el marcador GENERAR_RECOMENDACION_JSON: después de cada actividad que recomiendes
2. El formato es OBLIGATORIO - sin el JSON, la recomendación no será procesada
3. Para cada recomendación (máximo 5), usa este formato EXACTO:

**Actividad:** [título de la actividad]
**Categoría:** [categoría]
[tu texto descriptivo aquí]
---
GENERAR_RECOMENDACION_JSON: {"actividad_id": [número del campo "id"], "razon": "[razón breve]", "puntuacion": 0.8}

EJEMPLO COMPLETO CORRECTO:
**Actividad:** Campeonato de Futbol 2
**Categoría:** deportiva
Esta actividad es ideal para ti porque coincide con tu interés en deportes.
---
GENERAR_RECOMENDACION_JSON: {"actividad_id": 2, "razon": "Coincide con tu hobby de Deporte", "puntuacion": 0.9}

REGLAS ABSOLUTAS:
- actividad_id DEBE ser el número exacto del campo "id" de una actividad de la lista de actividades disponibles
- DEBES incluir "GENERAR_RECOMENDACION_JSON:" en cada recomendación (sin esto, NO funcionará)
- NO uses bloques de código ```json``` ni ``` - escribe el JSON directamente
- El JSON debe estar en UNA LÍNEA después del marcador
- Verifica que el ID existe en la lista antes de usarlo
- Si no incluyes el JSON, la recomendación será IGNORADA"""
    
    return prompt

