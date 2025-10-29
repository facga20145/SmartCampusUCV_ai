from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ActividadInput(BaseModel):
    """Modelo de entrada para una actividad."""
    id: int
    categoria: str
    titulo: Optional[str] = None
    descripcion: Optional[str] = None
    fecha: Optional[str] = None
    lugar: Optional[str] = None
    nivelSostenibilidad: Optional[int] = Field(None, alias="nivel_sostenibilidad")

class PreferenciaInput(BaseModel):
    """Modelo de entrada para una preferencia del usuario."""
    categoria: str
    nivel_interes: Optional[int] = None

class RecomendacionRequest(BaseModel):
    """Modelo de solicitud de recomendaciones."""
    actividades: List[ActividadInput]
    preferencias: List[PreferenciaInput]
    usuario_id: int
    historial_participacion: Optional[List[Dict[str, Any]]] = []
    user_query: Optional[str] = None

class RecomendacionOutput(BaseModel):
    """Modelo de salida para una recomendación."""
    actividad_id: int
    titulo: str
    categoria: str
    razon: str
    puntuacion: float
    actividad: Dict[str, Any]

class RecomendacionResponse(BaseModel):
    """Modelo de respuesta de recomendaciones."""
    recomendaciones: List[RecomendacionOutput]
    response_text: Optional[str] = None
    error: Optional[str] = None

class StatusResponse(BaseModel):
    """Modelo de respuesta de estado."""
    status: str
    nlp: str
    model: Optional[str] = None

