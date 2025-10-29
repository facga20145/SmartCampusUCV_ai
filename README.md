# SmartCampus AI - Microservicio de Recomendaciones Inteligentes

Microservicio de IA para SmartCampusUCV que genera recomendaciones personalizadas de actividades sostenibles usando Ollama y procesamiento de lenguaje natural.

## Descripción

Este microservicio utiliza modelos de lenguaje local (Ollama) para analizar las preferencias del usuario y las actividades disponibles, generando recomendaciones personalizadas basadas en:
- Preferencias de categorías del usuario
- Historial de participación
- Nivel de sostenibilidad de las actividades
- Disponibilidad de actividades

---

## Requisitos

- Python 3.10 o superior
- Ollama instalado con un modelo compatible (ej: `qwen2.5:3b-instruct`, `llama3.2:3b`)
- Dependencias listadas en `requirements.txt`

---

## Instalación

### 1. Crear y activar entorno virtual

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Instalar y configurar Ollama

```bash
# Instalar Ollama desde https://ollama.ai

# Descargar un modelo (ejemplo con qwen2.5:3b-instruct)
ollama pull qwen2.5:3b-instruct

# Verificar que el modelo está instalado
ollama list
```

### 4. Configurar el archivo de configuración

El archivo `src/ai/config/config.json` se creará automáticamente con valores por defecto si no existe.

---

## Uso

### 1. Iniciar el servidor

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

> **Nota:** El servidor de Ollama se iniciará automáticamente en segundo plano si no está ya en ejecución.

### 2. Acceder a la API

- **API base:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Documentación interactiva:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Endpoints Principales

### **POST /recomendar**

Genera recomendaciones personalizadas de actividades basadas en preferencias del usuario.

**Cuerpo de la solicitud:**
```json
{
  "actividades": [
    {
      "id": 1,
      "categoria": "ambiental",
      "titulo": "Recolección de basura",
      "descripcion": "Actividad de limpieza del campus",
      "fecha": "2025-11-01",
      "lugar": "Plaza Central"
    }
  ],
  "preferencias": [
    {
      "categoria": "ambiental",
      "nivel_interes": 5
    }
  ],
  "usuario_id": 1,
  "historial_participacion": []
}
```

**Respuesta:**
```json
{
  "recomendaciones": [
    {
      "actividad_id": 1,
      "titulo": "Recolección de basura",
      "categoria": "ambiental",
      "razon": "Coincide con tu interés en actividades ambientales",
      "puntuacion": 0.95
    }
  ]
}
```

### **GET /status**

Verifica el estado del microservicio y módulos.

**Respuesta:**
```json
{
  "status": "online",
  "nlp": "online",
  "model": "qwen2.5:3b-instruct"
}
```

---

## Estructura del Proyecto

```
smartcampus_ai/
├── .gitignore
├── .venv/
├── requirements.txt
├── README.md
└── src/
    ├── ai/
    │   ├── __init__.py
    │   ├── config/
    │   │   └── config.json
    │   └── nlp/
    │       ├── __init__.py
    │       ├── nlp_core.py
    │       ├── ollama_manager.py
    │       ├── prompt_creator.py
    │       ├── prompt_loader.py
    │       ├── system_prompt.yaml
    │       └── activity_analyzer.py
    ├── api/
    │   ├── __init__.py
    │   ├── routes.py
    │   ├── schemas.py
    │   └── utils.py
    ├── main.py
    └── utils/
        ├── __init__.py
        ├── error_handler.py
        └── logger_config.py
```

---

## Configuración

El archivo `src/ai/config/config.json` permite personalizar:

- **assistant_name**: Nombre del asistente
- **language**: Idioma de las respuestas
- **model**: Configuración del modelo Ollama (nombre, temperatura, tokens)
- **recommendation_settings**: Configuración de recomendaciones (número máximo, umbral de puntuación)

---

## Integración con Backend NestJS

El backend de SmartCampusUCV debe hacer llamadas HTTP a este microservicio:

```typescript
// Ejemplo desde el backend
const response = await this.httpService.post('http://localhost:8000/recomendar', {
  actividades: actividadesParaIA,
  preferencias: preferenciasParaIA,
  usuario_id: usuarioId,
  historial_participacion: historial
});
```

---

## Desarrollo

Para ejecutar en modo desarrollo con recarga automática:

```bash
uvicorn src.main:app --reload
```

Para ejecutar pruebas:

```bash
pytest tests/
```

---

## Notas

- El microservicio se comunica con Ollama en `http://localhost:11434`
- Asegúrate de que Ollama esté instalado y el modelo descargado antes de iniciar
- Los logs se guardan automáticamente para debugging

