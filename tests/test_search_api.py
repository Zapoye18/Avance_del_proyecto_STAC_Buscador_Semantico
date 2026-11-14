from fastapi.testclient import TestClient

from app.main import app
from app.search import (
    DocumentSearchIndex,
    SearchDocument,
)
from app import search_api


client = TestClient(app)


def create_test_index():
    documents = [
        SearchDocument(
            document_id="DOC-001",
            filename="servidores.txt",
            department="ingenieria",
            title="Problemas del servidor",
            text="Problemas de latencia en servidores",
        ),
        SearchDocument(
            document_id="DOC-002",
            filename="animaciones.txt",
            department="audiovisual",
            title="Animaciones",
            text="Animaciones y efectos visuales",
        ),
    ]

    return DocumentSearchIndex(documents)


def test_search_endpoint(monkeypatch):
    monkeypatch.setattr(
        search_api,
        "get_search_index",
        create_test_index,
    )

    response = client.post(
        "/search",
        json={
            "query": "latencia servidores",
            "top_k": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["results"]
    assert (
        data["results"][0]["department"]
        == "ingenieria"
    )


def test_search_rejects_invalid_top_k():
    response = client.post(
        "/search",
        json={
            "query": "problemas del servidor",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_search_rejects_empty_query():
    response = client.post(
        "/search",
        json={
            "query": "",
            "top_k": 5,
        },
    )

    assert response.status_code == 422