import json
from pathlib import Path

import joblib
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer

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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from app.preprocessing import transform_model_text

from scripts.train import (
    load_dataset,
    N_SPLITS,
    RANDOM_STATE,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT = ROOT / "models/classifier"

MODEL_NAME = "svm_v1"


def create_model():
    """
    Pipeline completo:

    Texto preparado
        -> Limpieza adicional
        -> TF-IDF
        -> SVM
    """

    return Pipeline([
        (
            "admin_cleanup",
            FunctionTransformer(
                transform_model_text,
                validate=False,
            ),
        ),
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
            LinearSVC(
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def main():
    df = load_dataset()

    texts = df["text"]
    labels = df["label"]

    print("\n=== ENTRENAMIENTO SVM V1 ===")
    print(f"Documentos: {len(df)}")

    print("\nDocumentos por departamento:")
    print(
        labels.value_counts()
        .sort_index()
        .to_string()
    )

    model = create_model()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    predictions = cross_val_predict(
        model,
        texts,
        labels,
        cv=cv,
    )

    classes = sorted(labels.unique())

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
    }

    print("\n=== MÉTRICAS ===")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n=== REPORTE POR DEPARTAMENTO ===")

    print(
        classification_report(
            labels,
            predictions,
            labels=classes,
            zero_division=0,
        )
    )

    print("\n=== MATRIZ DE CONFUSIÓN ===")

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=classes,
    )

    print("Orden de las categorías:")
    print(classes)

    print(matrix)

    # Una vez evaluado, entrenamos el
    # modelo utilizando todos los datos.
    model.fit(texts, labels)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = OUTPUT / f"{MODEL_NAME}.joblib"

    joblib.dump(
        model,
        model_path,
    )

    report = {
        "model": MODEL_NAME,
        "algorithm": "LinearSVC",
        "dataset": "documents_body.csv",
        "documents": len(df),
        "classes": classes,
        "preprocessing": "configuration_C",
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "metrics": metrics,
    }

    report_path = OUTPUT / f"{MODEL_NAME}_metrics.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== ARCHIVOS GENERADOS ===")
    print(model_path)
    print(report_path)

    print("\nENTRENAMIENTO COMPLETADO")


if __name__ == "__main__":
    main()