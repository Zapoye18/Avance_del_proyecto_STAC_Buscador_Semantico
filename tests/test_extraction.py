import pytest
from app.extraction import extract_fields, extract_document


def test_corpus_style_header():
    doc = extract_fields('''ID: ING-01
Título: Arquitectura del servidor
Tipo de documento: ADR
Autor/Rol: Backend Engineer
Versión: 1.1 | Estado: Vigente | Fecha: 2026-11-11
Documentos relacionados: QAL-03
====================
1. OBJETIVO
El servidor valida tokens.''')
    assert doc.source_id == "ING-01"
    assert doc.title == "Arquitectura del servidor"
    assert doc.author_role == "Backend Engineer"
    assert doc.metadata["version"] == "1.1"
    assert doc.body.startswith("1. OBJETIVO")
    assert "Autor/Rol" not in doc.body


def test_free_text_kept():
    doc = extract_fields("Tenemos un problema de red.\nSe perdió la conexión.")
    assert doc.title is None and doc.author_role is None
    assert "problema de red" in doc.body


def test_header_without_separator():
    doc = extract_fields("Título: Incidencia\nResponsable: QA Analyst\n\nEl juego se cierra.")
    assert doc.title == "Incidencia"
    assert doc.author_role == "QA Analyst"
    assert doc.body == "El juego se cierra."


def test_unrecognized_body_heading_is_not_removed():
    doc = extract_fields("OBJETIVO: Registrar pruebas\nDetalles: el juego falla")
    assert "OBJETIVO: Registrar pruebas" in doc.body


def test_unsupported_extension(tmp_path):
    path = tmp_path / "x.pdf"
    path.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="Formato"):
        extract_document(path)


def test_empty_document():
    with pytest.raises(ValueError, match="no contiene texto"):
        extract_fields(" \n \n")