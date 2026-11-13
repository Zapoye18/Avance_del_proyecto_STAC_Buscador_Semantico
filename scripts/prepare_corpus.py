"""Prepara dos datasets supervisados sin modificar el corpus original.
"""

from pathlib import Path
import re

import pandas as pd

from app.extraction import extract_document
from app.preprocessing import prepare_for_classification


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
OUTPUT = ROOT / "data" / "training"

DEPARTMENTS = (
    "diseno", "ingenieria", "audiovisual",
    "calidad", "operacion", "negocio",
)
REFERENCE = re.compile(r"\b(?:DIS|ING|ART|QAL|PRO|NEG)-\d+\b", re.I)


def build_dataframe(corpus_dir: Path = CORPUS) -> pd.DataFrame:
    """Extrae y prepara los documentos; solo la carpeta aporta la etiqueta."""
    records = []

    for department in DEPARTMENTS:
        folder = corpus_dir / department
        if not folder.is_dir():
            raise FileNotFoundError(f"Falta la carpeta: {folder}")

        for path in sorted(folder.glob("*.txt")):
            doc = extract_document(path)
            body = prepare_for_classification(doc, include_title=False)
            title_body = prepare_for_classification(doc, include_title=True)

            records.append({
                "document_id": path.stem,
                "label": department,
                "body_text": body,
                "title_body_text": title_body,
            })

    df = pd.DataFrame.from_records(
        records,
        columns=["document_id", "label", "body_text", "title_body_text"],
    )
    validate_dataframe(df)
    return df.sort_values("document_id").reset_index(drop=True)


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validaciones para el corpus académico fijo de 48 documentos."""
    expected = set(DEPARTMENTS)
    if len(df) != 48:
        raise ValueError(f"Se esperaban 48 documentos; encontrados: {len(df)}")
    counts = df["label"].value_counts().to_dict()
    if set(counts) != expected or any(counts[label] != 8 for label in expected):
        raise ValueError(f"Categorías desbalanceadas o inesperadas: {counts}")

    for col in ("document_id", "label", "body_text", "title_body_text"):
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Hay valores vacíos en {col}")

    if df["document_id"].duplicated().any():
        raise ValueError("Hay document_id duplicados")
    if df["body_text"].duplicated().any():
        raise ValueError("Hay textos de cuerpo exactamente duplicados")

    for col in ("body_text", "title_body_text"):
        if df[col].str.contains(REFERENCE, regex=True).any():
            raise ValueError(f"Quedan identificadores administrativos en {col}")


def export_datasets(df: pd.DataFrame, output_dir: Path = OUTPUT) -> None:
    """Exporta ambas variantes en idéntico orden, conservando su trazabilidad."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, source_column in (
        ("documents_body.csv", "body_text"),
        ("documents_title_body.csv", "title_body_text"),
    ):
        subset = df[["document_id", source_column, "label"]].rename(
            columns={source_column: "text"}
        )
        subset.to_csv(output_dir / filename, index=False, encoding="utf-8")
        print(f"Generado: {output_dir / filename} ({len(subset)} filas)")


def main() -> None:
    df = build_dataframe()
    print("\nDistribución por departamento:")
    print(df["label"].value_counts().sort_index().to_string())
    print("\nCampos vacíos:")
    print(df.isna().sum().to_string())
    export_datasets(df)
    print("\nPREPARACIÓN COMPLETADA")


if __name__ == "__main__":
    main()