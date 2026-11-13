
from collections import Counter
from pathlib import Path

from app.extraction import extract_document
from app.preprocessing import prepare_for_classification


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/corpus"


def get_context(text, position, length, margin=100):
    """Muestra el texto alrededor de una coincidencia."""
    start = max(0, position - margin)
    end = min(len(text), position + length + margin)

    return text[start:end].replace("\n", " ")


def main():
    counts = Counter()
    affected = 0

    for path in sorted(CORPUS.rglob("*.txt")):
        doc = extract_document(path)
        text = prepare_for_classification(doc)

        role = doc.author_role

        if not role:
            continue

        position = text.casefold().find(role.casefold())

        if position == -1:
            continue

        affected += 1

        print("\n" + "=" * 65)
        print("DOCUMENTO:", path.name)
        print("DEPARTAMENTO:", path.parent.name)
        print("ROL DETECTADO:", role)

        # Identificar dónde aparece en el
        # contenido extraído.
        body_position = doc.body.casefold().find(
            role.casefold()
        )

        if body_position >= 0:
            print("ORIGEN: aparece en doc.body")
            counts["Presente en cuerpo extraído"] += 1

            print(
                "CONTEXTO ORIGINAL:",
                get_context(
                    doc.body,
                    body_position,
                    len(role)
                )
            )
        else:
            print(
                "ORIGEN: incorporado durante "
                "el preprocesamiento"
            )
            counts["Agregado en preprocesamiento"] += 1

        print(
            "CONTEXTO CLASIFICACIÓN:",
            get_context(text, position, len(role))
        )

    print("\n" + "=" * 65)
    print("RESUMEN")
    print("Documentos afectados:", affected)
    print("Tipos de origen:", dict(counts))


if __name__ == "__main__":
    main()
