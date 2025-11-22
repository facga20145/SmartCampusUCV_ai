import requests
import json

url = "http://127.0.0.1:8000/recomendar"

payload = {
    "usuario_id": 1,
    "actividades": [
        {
            "id": 1,
            "categoria": "deportiva",
            "titulo": "Fútbol Sala",
            "descripcion": "Torneo de fútbol sala",
            "fecha": "2025-11-30",
            "lugar": "Cancha Principal",
            "nivel_sostenibilidad": 5
        },
        {
            "id": 2,
            "categoria": "cultural",
            "titulo": "Taller de Arte",
            "descripcion": "Aprende a pintar",
            "fecha": "2025-12-01",
            "lugar": "Aula Magna",
            "nivel_sostenibilidad": 8
        }
    ],
    "preferencias": [
        {"categoria": "deportiva", "nivel_interes": 5}
    ],
    "historial_participacion": [],
    "user_query": "Recomiéndame actividades deportivas"
}

print("Enviando petición a la IA...")
try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Respuesta:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Error: {e}")
