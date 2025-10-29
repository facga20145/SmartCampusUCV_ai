import logging
import sys
from pathlib import Path

def setup_logging():
    """Configura el sistema de logging para la aplicación."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = logging.INFO
    
    # Configurar logging a consola
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Sistema de logging configurado correctamente.")

