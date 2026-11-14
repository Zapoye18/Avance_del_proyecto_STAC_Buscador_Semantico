import re
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline

from scripts.train import (
    load_dataset,
    get_classifiers,
    N_SPLITS,
    RANDOM_STATE,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/evaluation"


# Lista deliberadamente pequeña.
# Conservamos las negaciones.
STOPWORDS = [
    "de", "del", "la", "las", "el", "los",
    "un", "una", "unos", "unas",
    "en", "con", "por", "para",
    "que", "se", "su", "sus",
    "al", "es", "son", "como",
]


# Se incluyen únicamente roles conocidos
# para este experimento.
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


# Fechas completas: 2026-11-13,
# 2026/11/13 o 2026-11.
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


def remove_admin_clues(text):
    """
    Elimina algunas pistas potencialmente
    administrativas para el experimento.

    No modifica los documentos originales.
    """

    text = DATE_PATTERN.sub(" ", text)
    text = ROLE_PATTERN.sub(" ", text)

    return re.sub(r"\s+", " ", text).strip()


def create_experimental_pipeline(
    classifier,
    remove_stopwords=False,
):
    """
    Construye un pipeline TF-IDF + modelo.
    """

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        norm="l2",
        stop_words=(
            STOPWORDS
            if remove_stopwords
            else None
        ),
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])


def main():
    df = load_dataset()

    original_texts = df["text"]

    cleaned_texts = original_texts.map(
        remove_admin_clues
    )

    if cleaned_texts.eq("").any():
        raise ValueError(
            "La eliminación de pistas produjo "
            "documentos vacíos."
        )

    labels = df["label"]

    experiments = {
        "A_original": {
            "texts": original_texts,
            "stopwords": False,
        },

        "B_sin_stopwords": {
            "texts": original_texts,
            "stopwords": True,
        },

        "C_sin_pistas_admin": {
            "texts": cleaned_texts,
            "stopwords": False,
        },

        "D_combinado": {
            "texts": cleaned_texts,
            "stopwords": True,
        },
    }

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for experiment_name, config in experiments.items():

        print("\n" + "=" * 65)
        print(f"EXPERIMENTO: {experiment_name}")
        print("=" * 65)

        for model_name, classifier in get_classifiers().items():

            pipeline = create_experimental_pipeline(
                classifier,
                remove_stopwords=config["stopwords"],
            )

            predictions = cross_val_predict(
                pipeline,
                config["texts"],
                labels,
                cv=cv,
            )

            metrics = {
                "experimento": experiment_name,
                "modelo": model_name,

                "accuracy": accuracy_score(
                    labels,
                    predictions,
                ),

                "f1_macro": f1_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),

                "recall_macro": recall_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
            }

            for department in sorted(labels.unique()):
                mask = labels == department

                metrics[f"recall_{department}"] = recall_score(
                    labels[mask],
                    pd.Series(predictions, index=labels.index)[mask],
                    labels=[department],
                    average="macro",
                    zero_division=0,
                )

            results.append(metrics)

            print(
                f"{model_name:15} "
                f"accuracy={metrics['accuracy']:.4f} "
                f"f1={metrics['f1_macro']:.4f}"
            )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("COMPARACIÓN FINAL")
    print("=" * 65)

    summary = results_df[
        [
            "experimento",
            "modelo",
            "accuracy",
            "f1_macro",
            "recall_macro",
        ]
    ].sort_values(
        ["modelo", "experimento"]
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT / "preprocessing_comparison.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nResultados guardados: {output_path}")


if __name__ == "__main__":
    main()