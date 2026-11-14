from app.preprocessing import (
    clean_model_text,
    transform_model_text,
)


def test_removes_date():
    text = (
        "El despliegue será el 2026-11-13."
    )

    result = clean_model_text(text)

    assert "2026-11-13" not in result
    assert "despliegue" in result


def test_removes_role():
    text = (
        "El Technical Artist revisó "
        "las texturas."
    )

    result = clean_model_text(text)

    assert "Technical Artist" not in result
    assert "texturas" in result


def test_preserves_negation():
    text = (
        "El jefe no recibe daño de fuego."
    )

    result = clean_model_text(text)

    assert "no recibe daño" in result


def test_preserves_technical_numbers():
    text = (
        "La textura utiliza 2048 px "
        "y consume 8 MB."
    )

    result = clean_model_text(text)

    assert "2048" in result
    assert "8 MB" in result


def test_transforms_multiple_documents():
    texts = [
        "Producer aprobó el despliegue.",
        "El sonido requiere 8 MB.",
    ]

    results = transform_model_text(texts)

    assert len(results) == 2
    assert "Producer" not in results[0]
    assert "8 MB" in results[1]