import pandas as pd
import pytest

from scripts.train import (
    DEPARTMENTS,
    N_SPLITS,
    load_dataset,
)


def make_dataset(counts):
    rows = []

    for department, amount in counts.items():
        for index in range(amount):
            rows.append({
                "document_id": (
                    f"{department}_{index}"
                ),
                "text": (
                    f"Contenido de prueba {index} "
                    f"para {department}"
                ),
                "label": department,
            })

    return pd.DataFrame(rows)


def test_accepts_different_document_counts(
    tmp_path,
):
    counts = {
        "diseno": 8,
        "ingenieria": 12,
        "audiovisual": 6,
        "calidad": 9,
        "operacion": 5,
        "negocio": 15,
    }

    df = make_dataset(counts)

    path = tmp_path / "dataset.csv"

    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )

    result = load_dataset(path)

    assert len(result) == 55
    assert set(result["label"]) == DEPARTMENTS


def test_rejects_insufficient_documents(
    tmp_path,
):
    counts = {
        department: N_SPLITS
        for department in DEPARTMENTS
    }

    counts["diseno"] = N_SPLITS - 1

    df = make_dataset(counts)

    path = tmp_path / "dataset.csv"

    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="al menos",
    ):
        load_dataset(path)


def test_rejects_duplicate_ids(
    tmp_path,
):
    counts = {
        department: N_SPLITS
        for department in DEPARTMENTS
    }

    df = make_dataset(counts)

    df.loc[1, "document_id"] = (
        df.loc[0, "document_id"]
    )

    path = tmp_path / "dataset.csv"

    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="duplicados",
    ):
        load_dataset(path)