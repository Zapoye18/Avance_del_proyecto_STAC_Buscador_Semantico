import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data/training/documents_body.csv"
OUTPUT = ROOT / "models/classifier"

DEPARTMENTS = {
    "diseno",
    "ingenieria",
    "audiovisual",
    "calidad",
    "operacion",
    "negocio",
}

N_SPLITS = 4
RANDOM_STATE = 42


def load_dataset(path=DATASET):
    """
    Carga y valida los documentos etiquetados.

    No exige exactamente ocho documentos por
    departamento. Cada categoría puede crecer.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "document_id",
        "text",
        "label",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "El dataset está vacío."
        )

    for column in required:
        df[column] = df[column].str.strip()

        if df[column].eq("").any():
            raise ValueError(
                f"Hay valores vacíos en {column}."
            )

    if df["document_id"].duplicated().any():
        raise ValueError(
            "Existen document_id duplicados."
        )

    invalid_labels = (
        set(df["label"]) - DEPARTMENTS
    )

    if invalid_labels:
        raise ValueError(
            "Departamentos desconocidos: "
            f"{sorted(invalid_labels)}"
        )

    counts = df["label"].value_counts()

    missing_departments = (
        DEPARTMENTS - set(counts.index)
    )

    if missing_departments:
        raise ValueError(
            "Faltan departamentos: "
            f"{sorted(missing_departments)}"
        )

    # Validación dependiente del número
    # de particiones, no del corpus inicial.
    insufficient = counts[
        counts < N_SPLITS
    ]

    if not insufficient.empty:
        raise ValueError(
            "Cada departamento necesita al menos "
            f"{N_SPLITS} documentos para esta "
            "validación cruzada.\n"
            f"{insufficient.to_dict()}"
        )

    return df


def create_pipeline(classifier):
    """
    Crea un pipeline completo:
    texto -> TF-IDF -> clasificador.

    El vectorizador se ajusta únicamente
    con los datos de entrenamiento de
    cada partición.
    """

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                norm="l2",
            ),
        ),
        (
            "classifier",
            classifier,
        ),
    ])


def get_classifiers():
    """
    Devuelve los tres algoritmos
    que vamos a comparar.
    """

    return {
        "naive_bayes": MultinomialNB(),

        "svm": LinearSVC(
            random_state=RANDOM_STATE,
        ),

        "logistic": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_model(
    name,
    classifier,
    texts,
    labels,
    cv,
):
    """
    Evalúa un modelo mediante predicciones
    fuera de muestra con validación cruzada.
    """

    pipeline = create_pipeline(classifier)

    predictions = cross_val_predict(
        pipeline,
        texts,
        labels,
        cv=cv,
    )

    classes = sorted(DEPARTMENTS)

    metrics = {
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),

        "precision_macro": float(
            precision_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "recall_macro": float(
            recall_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "f1_macro": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "confusion_matrix": (
            confusion_matrix(
                labels,
                predictions,
                labels=classes,
            ).tolist()
        ),

        "classes": classes,
    }

    print("\n" + "=" * 65)
    print(f"MODELO: {name.upper()}")
    print("=" * 65)

    print(
        classification_report(
            labels,
            predictions,
            labels=classes,
            zero_division=0,
        )
    )

    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=classes,
        columns=classes,
    )

    print("MATRIZ DE CONFUSIÓN")
    print("Filas: categoría real")
    print("Columnas: categoría predicha")
    print(matrix.to_string())

    # Entrenamiento final utilizando todos los
    # documentos etiquetados disponibles.
    pipeline.fit(texts, labels)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = OUTPUT / f"{name}.joblib"

    joblib.dump(
        pipeline,
        model_path,
    )

    print(f"\nModelo guardado: {model_path}")

    return metrics


def main():
    df = load_dataset()

    print("\n=== DATASET DE ENTRENAMIENTO ===")
    print(f"Documentos: {len(df)}")

    print("\nDOCUMENTOS POR DEPARTAMENTO")
    print(
        df["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    texts = df["text"]
    labels = df["label"]

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = {}

    for name, classifier in get_classifiers().items():
        results[name] = evaluate_model(
            name,
            classifier,
            texts,
            labels,
            cv,
        )

    comparison = pd.DataFrame([
        {
            "modelo": name,
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
        }
        for name, metrics in results.items()
    ])

    comparison = comparison.sort_values(
        by="f1_macro",
        ascending=False,
    )

    print("\n" + "=" * 65)
    print("COMPARACIÓN FINAL")
    print("=" * 65)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    winner = comparison.iloc[0]["modelo"]

    print(f"\nMEJOR F1-MACRO: {winner}")
    print(
        "Nota: resultado exploratorio; "
        "requiere revisión antes de publicar "
        "un modelo definitivo."
    )

    report = {
        "dataset": DATASET.name,
        "documents": len(df),
        "documents_per_department": (
            df["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "evaluation": {
            "method": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "vectorizer": {
            "type": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "strip_accents": "unicode",
            "norm": "l2",
        },
        "sklearn_version": sklearn.__version__,
        "results": results,
        "best_f1_macro": winner,
    }

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        OUTPUT / "comparison.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nResultados guardados en "
        "models/classifier/comparison.json"
    )


if __name__ == "__main__":
    main()