import logging
from fastapi import APIRouter, HTTPException
from typing import Optional
from api.schemas import (
    RecomendacionRequest,
    RecomendacionResponse,
    RecomendacionOutput,
    StatusResponse
)
from api import utils

logger = logging.getLogger("APIRoutes")

router = APIRouter()

@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Devuelve el estado actual del microservicio."""
    try:
        status = utils.get_module_status()
        return StatusResponse(**status)
    except Exception as e:
        logger.error(f"Error al obtener estado: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@router.post("/recomendar", response_model=RecomendacionResponse)
async def recomendar(request: RecomendacionRequest):
    """
    Genera recomendaciones personalizadas de actividades.
    
    Args:
        request: Solicitud con actividades, preferencias y datos del usuario
        
    Returns:
        Lista de recomendaciones generadas
    """
    try:
        nlp_module = utils.get_nlp_module()
        
        if not nlp_module:
            raise HTTPException(
                status_code=503,
                detail="El módulo NLP no está disponible"
            )
        
        if not nlp_module.is_online():
            raise HTTPException(
                status_code=503,
                detail="El módulo NLP está fuera de línea"
            )

        # Convertir actividades al formato esperado
        actividades_formato = []
        for act in request.actividades:
            actividades_formato.append({
                "id": act.id,
                "categoria": act.categoria,
                "titulo": act.titulo or "",
                "descripcion": act.descripcion or "",
                "fecha": act.fecha or "",
                "lugar": act.lugar or "",
                "nivel_sostenibilidad": act.nivelSostenibilidad
            })

        # Convertir preferencias al formato esperado
        preferencias_formato = []
        for pref in request.preferencias:
            preferencias_formato.append({
                "categoria": pref.categoria,
                "nivel_interes": pref.nivel_interes
            })

        # Generar recomendaciones
        result = await nlp_module.generate_recommendations(
            usuario_id=request.usuario_id,
            preferencias=preferencias_formato,
            actividades_disponibles=actividades_formato,
            historial_participacion=request.historial_participacion or [],
            hobbies=request.hobbies,
            intereses=request.intereses,
            user_query=request.user_query
        )

        if result.get("error"):
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )

        # Convertir a formato de respuesta
        recomendaciones_output = [
            RecomendacionOutput(**rec) for rec in result.get("recomendaciones", [])
        ]

        return RecomendacionResponse(
            recomendaciones=recomendaciones_output,
            response_text=result.get("response_text"),
            error=result.get("error")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al generar recomendaciones: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar recomendaciones: {str(e)}"
        )

