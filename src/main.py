import os
import asyncio
import logging
import warnings
import json
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
# Se busca en el directorio padre (smartcampus_ai) ya que main.py está en src
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# IMPORTANTE: Configurar la política del bucle de eventos ANTES de cualquier otra importación
if os.name == 'nt':  # Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from api import utils
from utils.logger_config import setup_logging
from utils.error_handler import ErrorHandler

setup_logging()
logger = logging.getLogger("MainApp")

app = FastAPI(
    title="SmartCampus AI API",
    description="Microservicio de IA para recomendaciones de actividades sostenibles",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = Path(__file__).parent / "ai" / "config" / "config.json"

@ErrorHandler.handle_exceptions
def load_config() -> Dict[str, Any]:
    """
    Carga la configuración desde config.json o crea una por defecto si no existe.
    
    Returns:
        Diccionario con la configuración cargada o por defecto.
    """
    default_config = {
        "assistant_name": "SmartCampus Assistant",
        "language": "es",
        "model": {
            "name": "llama-3.3-70b-versatile",
            "temperature": 0.7,
            "max_tokens": 1024
        },
        "recommendation_settings": {
            "max_recommendations": 5,
            "min_score": 0.3
        }
    }
    
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Configuración cargada desde {CONFIG_PATH}")
        else:
            logger.warning(f"Archivo de configuración no encontrado en {CONFIG_PATH}. Creando configuración por defecto.")
            config = default_config
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            logger.info(f"Configuración por defecto creada en {CONFIG_PATH}")
    except json.JSONDecodeError as e:
        logger.error(f"Error al decodificar JSON en {CONFIG_PATH}: {e}. Usando configuración por defecto.")
        config = default_config
    except Exception as e:
        logger.error(f"Error al cargar configuración: {e}. Usando configuración por defecto.")
        config = default_config
    
    return config

@app.on_event("startup")
@ErrorHandler.handle_async_exceptions
async def startup_event() -> None:
    """
    Evento de inicio de la aplicación.
    Inicializa la configuración y los módulos de IA.
    """
    logger.info("Iniciando SmartCampus AI...")
    
    config = load_config()
    logger.info(f"Configuración: {config}")
    
    utils.initialize_nlp_module(config)
    
    nlp_status = utils.get_module_status()
    if nlp_status["nlp"] == "online":
        logger.info("SmartCampus AI iniciado correctamente y en línea.")
    else:
        logger.warning("SmartCampus AI iniciado pero el módulo NLP no está en línea.")

@app.on_event("shutdown")
@ErrorHandler.handle_async_exceptions
async def shutdown_event() -> None:
    """
    Evento de cierre de la aplicación.
    """
    logger.info("Cerrando SmartCampus AI...")
    logger.info("SmartCampus AI cerrado correctamente.")

app.include_router(router, prefix="")

