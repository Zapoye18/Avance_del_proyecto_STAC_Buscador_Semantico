import argparse
from pathlib import Path

from app.extraction import extract_document
from app.preprocessing import (
    prepare_for_classification,
    prepare_for_search,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"


def show_document(path: Path):
    doc = extract_document(path)

    classification_body = prepare_for_classification(
        doc,
        include_title=False,
    )

    classification_title = prepare_for_classification(
        doc,
        include_title=True,
    )

    search_text = prepare_for_search(doc)

    print("\n" + "=" * 80)
    print(f"DOCUMENTO: {path.name}")
    print(f"DEPARTAMENTO: {path.parent.name}")
    print(f"TÍTULO EXTRAÍDO: {doc.title}")
    print(f"ROL EXTRAÍDO: {doc.author_role}")

    print("\n" + "-" * 80)
    print("1. CUERPO EXTRAÍDO (ORIGINAL)")
    print("-" * 80)
    print(doc.body)

    print("\n" + "-" * 80)
    print("2. CLASIFICACIÓN: SOLO CUERPO")
    print("-" * 80)
    print(classification_body)

    print("\n" + "-" * 80)
    print("3. CLASIFICACIÓN: TÍTULO + CUERPO")
    print("-" * 80)
    print(classification_title)

    print("\n" + "-" * 80)
    print("4. BÚSQUEDA: TÍTULO + CUERPO")
    print("-" * 80)
    print(search_text)

    print("\n" + "-" * 80)
    print("5. ESTADÍSTICAS")
    print("-" * 80)
    print(f"Caracteres originales: {len(doc.body)}")
    print(f"Caracteres clasificación: {len(classification_body)}")
    print(f"Caracteres búsqueda: {len(search_text)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--document",
        help="Nombre exacto de un documento .txt"
    )

    args = parser.parse_args()

    documents = sorted(CORPUS.rglob("*.txt"))

    if args.document:
        documents = [
            path for path in documents
            if path.name == args.document
        ]

        if not documents:
            parser.error(
                f"No se encontró: {args.document}"
            )

    for path in documents:
        show_document(path)

    print(f"\nDOCUMENTOS MOSTRADOS: {len(documents)}")


if __name__ == "__main__":
    main()
