from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rejects_unsupported_file():
    response = client.post(
        "/classify",
        files={
            "file": (
                "documento.pdf",
                b"contenido de prueba",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 415


def test_rejects_empty_file():
    response = client.post(
        "/classify",
        files={
            "file": (
                "documento.txt",
                b"",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400