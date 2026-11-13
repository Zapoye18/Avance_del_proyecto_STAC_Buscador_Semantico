from app.extraction import ExtractedDocument
from app.preprocessing import normalize_text, prepare_for_classification, prepare_for_search


from app.preprocessing import (
    normalize_text,
    remove_document_references,
    remove_administrative_fields,
    prepare_for_classification,
    prepare_for_search,
)


def test_normalization_keeps_negation():
    text = "El jefe  NO\nrecibe daño"

    result = normalize_text(text)

    assert result == "El jefe NO recibe daño"


def test_document_references_are_anonymized():
    text = "Consultar ING-02 y QAL-01."

    result = remove_document_references(text)

    assert "ING-02" not in result
    assert "QAL-01" not in result
    assert result.count(
        "DOCUMENTO_REFERENCIADO"
    ) == 2


def test_removes_explicit_author_assignment():
    text = (
        "El documento describe las texturas.\n"
        "Responsable: Technical Artist."
    )

    result = remove_administrative_fields(
        text,
        author_role="Technical Artist"
    )

    assert "Responsable: Technical Artist" not in result
    assert "describe las texturas" in result


def test_preserves_narrative_role():
    text = (
        "El Art Director aprueba cada asset."
    )

    result = remove_administrative_fields(
        text,
        author_role="Art Director"
    )

    assert "El Art Director aprueba" in result


def test_classification_excludes_header_role_and_reference():
    doc = ExtractedDocument(
        title="Diseño del jefe DIS-06",
        author_role="Lead Game Designer",
        document_type="Acta de diseño",
        source_id="DIS-06",
        body=(
            "La fase 2 remite a QAL-01.\n"
            "No recibe daño de fuego.\n"
            "Responsable: Lead Game Designer."
        ),
    )

    result = prepare_for_classification(doc)

    assert "DIS-06" not in result
    assert "QAL-01" not in result
    assert "Responsable: Lead Game Designer" not in result
    assert "No recibe daño de fuego" in result
    assert "Diseño del jefe" not in result

    titled = prepare_for_classification(
        doc,
        include_title=True
    )

    assert "Diseño del jefe" in titled
    assert "DIS-06" not in titled

    search_text = prepare_for_search(doc)

    assert "QAL-01" in search_text
    assert "DIS-06" in search_text
    assert "Lead Game Designer" in search_text


def test_search_preserves_document_references():
    doc = ExtractedDocument(
        title="Arquitectura del servidor",
        author_role="Backend Engineer",
        body="Consultar ING-02 y QAL-01."
    )

    result = prepare_for_search(doc)

    assert "ING-02" in result
    assert "QAL-01" in result
    assert "Arquitectura del servidor" in result


def test_classification_keeps_technical_content():
    doc = ExtractedDocument(
        title="Informe técnico",
        author_role="Backend Engineer",
        body=(
            "El servidor utiliza una API REST.\n"
            "Responsable: Backend Engineer."
        )
    )

    result = prepare_for_classification(doc)

    assert "API REST" in result
    assert "Responsable: Backend Engineer" not in result
