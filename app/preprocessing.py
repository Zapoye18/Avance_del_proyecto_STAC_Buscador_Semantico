from __future__ import annotations
import re
import unicodedata
from app.extraction import ExtractedDocument

# Identificadores propios del corpus
DOCUMENT_REF_PATTERN = re.compile(
    r"\b(?:DIS|ING|ART|QAL|PRO|NEG)-\d+\b",
    flags=re.IGNORECASE
)


def normalize_text(text: str) -> str:
    """
    Normaliza Unicode y espacios.

    No elimina negaciones ni palabras
    importantes para el significado.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_document_references(text: str) -> str:
    """
    Sustituye identificadores administrativos
    por una referencia genérica.

    Ejemplo:
    'Consultar ING-02'
    'Consultar DOCUMENTO_REFERENCIADO'
    """
    return DOCUMENT_REF_PATTERN.sub(
        "DOCUMENTO_REFERENCIADO",
        text
    )


def remove_administrative_fields(
    text: str,
    author_role: str | None = None
) -> str:
    """
    Elimina campos administrativos explícitos.

    Conserva las menciones narrativas
    a los roles profesionales.
    """
    # Quitar campos de cabecera que pudieran
    # permanecer dentro del cuerpo.
    text = re.sub(
        r"(?im)^\s*(?:Autor/Rol|Autor)\s*:"
        r"[^\r\n]*(?:\r?\n|$)",
        "",
        text
    )

    if author_role:
        # Eliminar únicamente una asignación
        # explícita al rol del autor.
        role = re.escape(author_role)

        pattern = re.compile(
            r"\bResponsable"
            r"(?:\s*\(owner\)|\s+de ejecución)?"
            r"\s*:\s*"
            + role
            + r"(?=\s*(?:[.;,)]|$))",
            flags=re.IGNORECASE
        )

        text = pattern.sub("", text)

    return text


def prepare_for_classification(
    document,
    include_title: bool = False
) -> str:
    """
    Prepara el texto para clasificación.

    Excluye identificadores y campos
    administrativos explícitos del autor.
    """
    text = remove_administrative_fields(
        document.body,
        author_role=document.author_role
    )

    if include_title and document.title:
        text = f"{document.title}\n{text}"

    text = remove_document_references(text)

    # Limpiar puntuación residual cuando
    # se elimina una asignación administrativa.
    # Eliminar espacios antes de puntuación.
    text = re.sub(r"\s+([.,;])", r"\1", text)

    # Corregir puntos consecutivos accidentales,
    # sin alterar los puntos suspensivos.
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)

    return normalize_text(text)


def prepare_for_search(document) -> str:
    """
    Prepara texto para búsqueda semántica.

    Conserva título, contenido y referencias
    documentales originales.
    """
    parts = []

    if document.title:
        parts.append(document.title)

    parts.append(document.body)

    return normalize_text("\n".join(parts))
