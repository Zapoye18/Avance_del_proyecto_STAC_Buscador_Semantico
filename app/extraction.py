from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re
import unicodedata

_SEPARATOR = re.compile(r"^\s*={10,}\s*$")
_FIELD = re.compile(r"^\s*([^:]+?)\s*:\s*(.*?)\s*$")

_FIELD_ALIASES = {
    "id": "source_id",
    "titulo": "title",
    "autor/rol": "author_role",
    "autor": "author_role",
    "responsable": "author_role",
    "tipo de documento": "document_type",
    "version": "version",
    "estado": "status",
    "fecha": "date",
    "proyecto": "project",
    "documentos relacionados": "related_documents",
}


def _canonical_key(key: str) -> str:
    normalized = unicodedata.normalize("NFKD", key.casefold())
    return "".join(c for c in normalized if not unicodedata.combining(c)).strip()


@dataclass(frozen=True)
class ExtractedDocument:
    body: str
    title: str | None = None
    author_role: str | None = None
    source_id: str | None = None
    document_type: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    extraction_warnings: tuple[str, ...] = ()


def _parse_header(lines: list[str]) -> dict[str, str]:
    """Parse known metadata in a header, including pipe-separated metadata."""
    result: dict[str, str] = {}
    for line in lines:
        for part in line.split(" | "):
            match = _FIELD.fullmatch(part)
            if not match:
                continue
            key = _FIELD_ALIASES.get(_canonical_key(match.group(1)))
            if key and match.group(2).strip():
                result[key] = match.group(2).strip()
    return result


def extract_fields(text: str) -> ExtractedDocument:
    """Extract known metadata; preserve main body, even in free-form TXT.

    Structured corpus: known header, followed by a line of >=10 '='.
    Free form: consecutive recognized metadata lines at beginning, followed
    by text. Unrecognized lines and section headings are kept as body.
    """
    if not isinstance(text, str):
        raise TypeError("Se esperaba texto (str)")
    normalized_lines = text.lstrip("\ufeff").splitlines()
    if not any(line.strip() for line in normalized_lines):
        raise ValueError("El documento no contiene texto")

    warnings: list[str] = []
    separator_at = next(
        (i for i, line in enumerate(normalized_lines[:35])
         if _SEPARATOR.fullmatch(line)),
        None,
    )
    if separator_at is not None:
        possible_header = _parse_header(normalized_lines[:separator_at])
        if possible_header:
            metadata = possible_header
            body_lines = normalized_lines[separator_at + 1:]
        else:
            metadata = {}
            body_lines = normalized_lines
            warnings.append("Separador sin cabecera reconocible; texto conservado")
    else:
        metadata = {}
        start = 0
        while start < len(normalized_lines) and not normalized_lines[start].strip():
            start += 1
        i = start
        recognized_lines = 0
        while i < len(normalized_lines):
            line = normalized_lines[i]
            if not line.strip() and recognized_lines:
                i += 1
                break
            new = _parse_header([line])
            if not new:
                break
            metadata.update(new)
            recognized_lines += 1
            i += 1
        body_lines = normalized_lines[i:] if recognized_lines else normalized_lines
        if recognized_lines:
            warnings.append("Cabecera sin separador: se reconocieron solo campos conocidos")
        else:
            warnings.append("Texto libre sin cabecera reconocible")

    body = "\n".join(body_lines).strip()
    if not body:
        raise ValueError("No se encontró contenido sustantivo")
    return ExtractedDocument(
        body=body,
        title=metadata.get("title"),
        author_role=metadata.get("author_role"),
        source_id=metadata.get("source_id"),
        document_type=metadata.get("document_type"),
        metadata=metadata,
        extraction_warnings=tuple(warnings),
    )


def extract_document(path: Path) -> ExtractedDocument:
    """Read a TXT. PDF/DOCX readers will be added in subsequent stages."""
    path = Path(path)
    if path.suffix.lower() != ".txt":
        raise ValueError(f"Formato no admitido todavía: {path.suffix}")
    try:
        contents = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("El archivo TXT no está codificado en UTF-8") from exc
    return extract_fields(contents)