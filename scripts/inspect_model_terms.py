from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models/classifier"

MODEL_NAMES = [
    "naive_bayes",
    "svm",
    "logistic",
]

TOP_N = 15


def inspect_model(name):
    path = MODELS / f"{name}.joblib"

    if not path.exists():
        print(f"No se encontró: {path}")
        return

    pipeline = joblib.load(path)

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    terms = vectorizer.get_feature_names_out()
    classes = classifier.classes_

    if name == "naive_bayes":
        weights = classifier.feature_log_prob_
    else:
        weights = classifier.coef_

    print("\n" + "=" * 65)
    print(f"MODELO: {name.upper()}")
    print("=" * 65)

    for index, department in enumerate(classes):
        top_indices = np.argsort(
            weights[index]
        )[-TOP_N:][::-1]

        result = pd.DataFrame({
            "termino": terms[top_indices],
            "peso": weights[index][top_indices],
        })

        print(f"\nDEPARTAMENTO: {department}")
        print(
            result.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )


def main():
    for name in MODEL_NAMES:
        inspect_model(name)


if __name__ == "__main__":
    main()