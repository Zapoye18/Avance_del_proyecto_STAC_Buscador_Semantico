"""Pruebas del contrato de validación del DataFrame."""

import pandas as pd
import pytest

from scripts.prepare_corpus import DEPARTMENTS, validate_dataframe


def valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "document_id": f"DOC-{department}-{i}",
            "label": department,
            "body_text": f"Contenido {department} {i}",
            "title_body_text": f"Título {department} {i}. Contenido {department} {i}",
        }
        for department in DEPARTMENTS for i in range(8)
    ])


def test_valid_corpus():
    validate_dataframe(valid_dataframe())


def test_duplicate_document_id():
    df = valid_dataframe()
    df.loc[1, "document_id"] = df.loc[0, "document_id"]
    with pytest.raises(ValueError, match="document_id duplicados"):
        validate_dataframe(df)


def test_administrative_reference():
    df = valid_dataframe()
    df.loc[0, "body_text"] = "Consultar ING-02"
    with pytest.raises(ValueError, match="identificadores administrativos"):
        validate_dataframe(df)


def test_missing_text():
    df = valid_dataframe()
    df.loc[0, "body_text"] = " "
    with pytest.raises(ValueError, match="valores vacíos"):
        validate_dataframe(df)