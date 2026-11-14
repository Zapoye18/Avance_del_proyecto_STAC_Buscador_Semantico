from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)

from scripts.train import (
    RANDOM_STATE,
    N_SPLITS,
    create_pipeline,
    get_classifiers,
    load_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/evaluation"


def main():
    df = load_dataset()

    texts = df["text"]
    labels = df["label"]

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = df[
        ["document_id", "label"]
    ].copy()

    print("\n=== ANÁLISIS DE ERRORES ===")

    for name, classifier in get_classifiers().items():
        pipeline = create_pipeline(classifier)

        predictions = cross_val_predict(
            pipeline,
            texts,
            labels,
            cv=cv,
        )

        results[name] = predictions

        errors = results[
            results["label"] != results[name]
        ]

        print("\n" + "=" * 65)
        print(f"MODELO: {name.upper()}")
        print("=" * 65)

        print(f"Errores: {len(errors)}")

        if not errors.empty:
            print(
                errors[
                    ["document_id", "label", name]
                ].to_string(index=False)
            )

    # Identificar documentos donde los
    # clasificadores no coinciden.
    model_columns = list(
        get_classifiers().keys()
    )

    disagreement = results[
        results[model_columns].nunique(axis=1) > 1
    ]

    print("\n" + "=" * 65)
    print("DESACUERDOS ENTRE MODELOS")
    print("=" * 65)

    print(
        f"Documentos con predicciones "
        f"diferentes: {len(disagreement)}"
    )

    if not disagreement.empty:
        print(
            disagreement.to_string(
                index=False
            )
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT / "classification_predictions.csv"
    )

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nResultados guardados en: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()