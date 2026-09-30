"""Publica las fichas de entrenamiento en Gestión Documental."""

from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path

import requests


DEFAULT_API_URL = "https://monitor-backend-annar-dev-evhaehfub5a8g4d3.eastus2-01.azurewebsites.net"
PROJECT_TOPICS = {
    "lis_yolox": "Pruebas rapidas lis",
    "vet_yolox": "veterinaria",
}
TIMEOUT = 30


def _json(response, operation):
    try:
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"Gestión Documental: falló {operation} (HTTP {response.status_code})") from exc


def publish_reports(report_dir: Path, project: str, version: str) -> None:
    """Autentica, asegura la categoría del proyecto y sube HTML y Excel."""
    topic_name = PROJECT_TOPICS.get(project)
    if topic_name is None:
        raise ValueError(f"No hay categoría documental definida para {project}")
    client_id = os.getenv("DOCUMENT_API_CLIENT_ID", "").strip()
    client_secret = os.getenv("DOCUMENT_API_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise ValueError("Faltan DOCUMENT_API_CLIENT_ID o DOCUMENT_API_CLIENT_SECRET")
    base_url = os.getenv("DOCUMENT_API_BASE_URL", DEFAULT_API_URL).strip().rstrip("/")
    if not base_url.startswith("https://"):
        raise ValueError("DOCUMENT_API_BASE_URL debe usar HTTPS")
    paths = [report_dir / "model_report.html", report_dir / "model_report.xlsx"]
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"No existe el informe documental: {path}")

    with requests.Session() as session:
        token_data = _json(session.post(
            f"{base_url}/api/external/auth/token",
            json={"clientId": client_id, "clientSecret": client_secret}, timeout=TIMEOUT,
        ), "la autenticación")
        token = token_data.get("accessToken") if isinstance(token_data, dict) else None
        if not token:
            raise RuntimeError("Gestión Documental no devolvió accessToken")
        session.headers.update({"Authorization": f"Bearer {token}"})
        topics_url = f"{base_url}/api/external/documents/topics"
        topics = _json(session.get(topics_url, timeout=TIMEOUT), "la consulta de categorías")
        if not isinstance(topics, list):
            raise RuntimeError("Gestión Documental devolvió categorías en un formato inesperado")
        topic = next((item for item in topics if isinstance(item, dict)
                      and item.get("name") == topic_name), None)
        if topic is None:
            response = session.post(topics_url, json={
                "name": topic_name, "description": f"Informes del proyecto {project}",
            }, timeout=TIMEOUT)
            if response.status_code == 409:
                topics = _json(session.get(topics_url, timeout=TIMEOUT), "la consulta de categorías")
                topic = next((item for item in topics if isinstance(item, dict)
                              and item.get("name") == topic_name), None)
                if topic is None:
                    raise RuntimeError(f"La categoría {topic_name} existe pero no se pudo consultar")
            else:
                topic = _json(response, "la creación de categoría")
        topic_id = topic.get("id") if isinstance(topic, dict) else None
        if not topic_id:
            raise RuntimeError(f"La categoría {topic_name} no devolvió id")

        with ExitStack() as stack:
            files = [("file", (f"{project}_{version}_{path.name}",
                                stack.enter_context(path.open("rb")),
                                "text/html" if path.suffix == ".html" else
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
                     for path in paths]
            result = _json(session.post(
                f"{base_url}/api/external/documents",
                data={"topicId": topic_id}, files=files, timeout=120,
            ), "la carga de documentos")
        if not isinstance(result, list) or len(result) != len(paths) or any(
            not isinstance(item, dict) or item.get("success") is not True for item in result
        ):
            raise RuntimeError(f"Gestión Documental no confirmó ambos documentos: {result!r}")
