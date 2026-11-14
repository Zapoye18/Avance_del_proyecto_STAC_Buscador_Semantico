from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT / "data"
    / "training"
    / "documents_body.csv"
)


def load_dataset():
    """Carga el dataset preparado con Pandas."""

    df = pd.read_csv(
        DATASET,
        encoding="utf-8-sig"
    )

    required = {
        "document_id",
        "text",
        "label"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "Faltan columnas obligatorias"
        )

    if df[list(required)].isna().any().any():
        raise ValueError(
            "El dataset contiene valores vacíos"
        )

    return df


def vectorize_documents(df):
    """Convierte textos en una matriz TF-IDF."""

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_df=1.0,
        min_df=1,
        norm="l2"
    )

    matrix = vectorizer.fit_transform(
        df["text"]
    )

    return vectorizer, matrix


def show_results(df, vectorizer, matrix):
    """Muestra estadísticas y características."""

    vocabulary = vectorizer.get_feature_names_out()

    print("\n=== MATRIZ TF-IDF ===")
    print("Documentos:", matrix.shape[0])
    print("Características:", matrix.shape[1])
    print("Valores no nulos:", matrix.nnz)

    print("\n=== VOCABULARIO ===")
    print("Primeros 30 términos:")
    print(vocabulary[:30])

    print("\n=== PRIMER DOCUMENTO ===")

    first_vector = matrix[0].toarray()[0]

    weights = pd.DataFrame({
        "term": vocabulary,
        "tfidf": first_vector
    })

    weights = (
        weights[weights["tfidf"] > 0]
        .sort_values(
            "tfidf",
            ascending=False
        )
        .head(15)
    )

    print("Documento:", df.iloc[0]["document_id"])
    print("Departamento:", df.iloc[0]["label"])

    print("\nTérminos con mayor peso:")
    print(weights.to_string(index=False))


def main():
    df = load_dataset()

    vectorizer, matrix = (
        vectorize_documents(df)
    )

    show_results(
        df,
        vectorizer,
        matrix
    )


if __name__ == "__main__":
    main()
