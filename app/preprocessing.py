from __future__ import annotations
import re
import pandas as pd
import unicodedata
from app.extraction import ExtractedDocument

ROLES = [
    "lead game designer",
    "game designer",
    "level designer",
    "systems designer",
    "art director",
    "technical artist",
    "3d artist",
    "sound designer",
    "audio director",
    "qa analyst",
    "qa lead",
    "qa tester",
    "backend engineer",
    "gameplay programmer",
    "engine programmer",
    "technical lead",
    "devops engineer",
    "release manager",
    "associate producer",
    "producer",
    "project manager",
    "product manager",
    "product analyst",
    "marketing manager",
    "community manager",
    "support agent",
    "support lead",
]


DATE_PATTERN = re.compile(
    r"\b20\d{2}[-/](?:0?[1-9]|1[0-2])"
    r"(?:[-/](?:0?[1-9]|[12]\d|3[01]))?\b"
)


ROLE_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(role)
        for role in sorted(
            ROLES,
            key=len,
            reverse=True,
        )
    )
    + r")\b",
    flags=re.IGNORECASE,
)

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

def clean_model_text(text):
    """
    Elimina fechas y menciones de roles
    seleccionadas para la configuración C.

    Conserva el resto del contenido.
    """

    text = str(text)

    text = DATE_PATTERN.sub(
        " ",
        text,
    )

    text = ROLE_PATTERN.sub(
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def transform_model_text(texts):
    """
    Recibe una colección de textos
    y devuelve los textos preparados.

    Se utiliza dentro del Pipeline.
    """

    return pd.Series(texts).map(
        clean_model_text
    ).tolist()