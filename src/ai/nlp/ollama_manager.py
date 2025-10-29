import subprocess
import time
import logging
import os
import ollama
from typing import Dict, Any, Optional

logger = logging.getLogger("OllamaManager")

class OllamaManager:
    """Gestiona el ciclo de vida del servidor Ollama y la conectividad del modelo."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Inicializa OllamaManager, intentando iniciar el servidor Ollama y verificar la conexión.

        Args:
            model_config: Configuración del modelo Ollama (nombre, temperatura, max_tokens)
        """
        self._ollama_process: Optional[subprocess.Popen] = None
        self._online: bool = False
        self._model_config: Dict[str, Any] = model_config
        self._model_name: Optional[str] = model_config.get("name")
        logger.debug("Iniciando Ollama server...")
        self._start_ollama_server()
        logger.debug("Verificando conexión a Ollama...")
        self._online = self._check_connection()
        if self._online:
            logger.info("OllamaManager inicializado y en línea.")
        else:
            logger.warning("OllamaManager inicializado pero no está en línea.")

    def _start_ollama_server(self, retries: int = 30, delay: int = 1):
        """Inicia el servidor de Ollama como un subproceso si no está ya en ejecución."""
        logger.debug("Verificando si el servidor Ollama ya está en ejecución.")
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        try:
            client = ollama.Client(host=ollama_host)
            client.list()
            logger.info("El servidor de Ollama ya está en ejecución.")
            self._online = True
            return
        except ollama.ResponseError:
            logger.info("El servidor de Ollama no está en ejecución, intentando iniciarlo...")
        except Exception as e:
            logger.warning(f"Error al verificar el estado de Ollama: {e}. Intentando iniciar el servidor.")

        try:
            self._ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            logger.info("Servidor de Ollama iniciado en segundo plano.")
            
            for attempt in range(retries):
                try:
                    client = ollama.Client(host=ollama_host)
                    client.list()
                    logger.info("Conexión con el servidor de Ollama establecida exitosamente.")
                    self._online = True
                    return
                except ollama.ResponseError:
                    logger.debug(f"Intento {attempt + 1}/{retries}: Servidor Ollama aún no disponible. Reintentando en {delay}s...")
                    time.sleep(delay)
            
            logger.error("Fallo al conectar con el servidor de Ollama después de iniciarlo.")
            self._online = False

        except FileNotFoundError:
            logger.error("El comando 'ollama' no se encontró. Asegúrese de que Ollama esté instalado y en el PATH.")
            self._online = False
        except Exception as e:
            logger.error(f"Error inesperado al iniciar el servidor de Ollama: {e}")
            self._online = False

    def __del__(self):
        """Asegura que el proceso de Ollama se termine al cerrar la aplicación."""
        self.close()

    def close(self):
        """Termina explícitamente el proceso del servidor de Ollama si está en ejecución."""
        if self._ollama_process and self._ollama_process.poll() is None:
            logger.info("Terminando el proceso del servidor de Ollama...")
            try:
                self._ollama_process.terminate()
                self._ollama_process.wait(timeout=5)
                if self._ollama_process.poll() is None:
                    logger.warning("El proceso de Ollama no terminó, forzando el cierre...")
                    self._ollama_process.kill()
                    self._ollama_process.wait()
                logger.info(f"Proceso del servidor de Ollama terminado.")
            except Exception as e:
                logger.error(f"Error al intentar terminar el proceso de Ollama: {e}")
            finally:
                self._ollama_process = None

    def _check_connection(self) -> bool:
        """Verifica la conexión con Ollama y la disponibilidad del modelo configurado."""
        logger.debug("Realizando verificación de conexión y modelo Ollama.")
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        try:
            client = ollama.Client(host=ollama_host)
            available_models = client.list()
            model_names = [m['name'] for m in available_models.get('models', [])]
            logger.debug(f"Modelos Ollama disponibles: {', '.join(model_names)}")

            if not self._model_name or self._model_name not in model_names:
                logger.error(f"Error: El modelo '{self._model_name}' no está disponible.")
                logger.error("Modelos disponibles: " + ", ".join(model_names) if model_names else "Ninguno.")
                return False
            
            logger.info(f"Conexión Ollama y modelo '{self._model_name}' verificados exitosamente.")
            return True
        except Exception as e:
            logger.error(f"Error al verificar la conexión de Ollama: {e}")
            return False

    def is_online(self) -> bool:
        """Indica si el módulo Ollama está en línea y listo para usarse."""
        return self._online

    def reload(self, model_config: Dict[str, Any]):
        """Recarga la configuración del modelo y revalida la conexión con Ollama."""
        logger.info("Recargando configuración de Ollama y revalidando conexión...")
        self._model_config = model_config
        self._model_name = model_config.get("name")
        self._online = self._check_connection()
        if self._online:
            logger.info("Ollama recargado y en línea.")
        else:
            logger.warning("Ollama recargado pero no está en línea.")

