import argparse
from pathlib import Path

from app.classification import classify_document


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clasifica un documento utilizando SVM v1."
        )
    )

    parser.add_argument(
        "--document",
        required=True,
        help="Ruta del documento que se clasificará.",
    )

    args = parser.parse_args()

    path = Path(args.document)

    if not path.is_absolute():
        path = ROOT / path

    result = classify_document(path)

    print("\n=== RESULTADO DE CLASIFICACIÓN ===")

    print(
        f"Documento: {result['filename']}"
    )

    print(
        f"Departamento: "
        f"{result['predicted_department']}"
    )

    print(
        f"Modelo: {result['model']}"
    )


if __name__ == "__main__":
    main()