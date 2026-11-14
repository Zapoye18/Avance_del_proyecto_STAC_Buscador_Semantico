import pytest

from app.classification import (
    classify_text,
    DEPARTMENTS,
)


class FakeModel:
    """
    Simula un modelo entrenado.

    Permite comprobar la lógica del
    módulo sin cargar archivos .joblib.
    """

    def predict(self, texts):
        return ["ingenieria"]


def test_classify_text():
    model = FakeModel()

    result = classify_text(
        "El servidor presenta problemas de latencia.",
        model=model,
    )

    assert result == "ingenieria"


def test_rejects_empty_text():
    model = FakeModel()

    with pytest.raises(
        ValueError,
        match="vacío",
    ):
        classify_text(
            "   ",
            model=model,
        )


def test_rejects_invalid_type():
    model = FakeModel()

    with pytest.raises(TypeError):
        classify_text(
            None,
            model=model,
        )


def test_returns_valid_department():
    model = FakeModel()

    result = classify_text(
        "Se detectó un problema de conexión.",
        model=model,
    )

    assert result in DEPARTMENTS