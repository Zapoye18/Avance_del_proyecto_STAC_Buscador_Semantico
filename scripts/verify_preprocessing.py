import re
from collections import Counter
from pathlib import Path

from app.extraction import extract_document

from app.preprocessing import (
    prepare_for_classification,
    prepare_for_search,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/corpus"


REFERENCE_PATTERN = re.compile(
    r"\b(?:DIS|ING|ART|QAL|PRO|NEG)-\d+\b",
    flags=re.IGNORECASE
)


def main():
    documents = sorted(CORPUS.rglob("*.txt"))

    problems = []
    counts = Counter()

    if not documents:
        raise RuntimeError(
            "No se encontraron documentos"
        )

    for path in documents:
        doc = extract_document(path)

        classification_text = (
            prepare_for_classification(doc)
        )

        search_text = prepare_for_search(doc)

        counts[path.parent.name] += 1

        # 1. Identificadores administrativos
        if REFERENCE_PATTERN.search(
            classification_text
        ):
            problems.append(
                f"{path.name}: referencia sin anonimizar"
            )

        # 2. Asignación explícita al autor
        if doc.author_role:
            role = re.escape(doc.author_role)

            assignment_pattern = re.compile(
                r"\bResponsable"
                r"(?:\s*\(owner\)|\s+de ejecución)?"
                r"\s*:\s*"
                + role
                + r"(?=\s*(?:[.;,)]|$))",
                flags=re.IGNORECASE
            )

            if assignment_pattern.search(
                classification_text
            ):
                problems.append(
                    f"{path.name}: asignación administrativa"
                )

        # 3. Título en modo solo cuerpo
        if (
            doc.title
            and classification_text.startswith(
                doc.title
            )
        ):
            problems.append(
                f"{path.name}: título incorporado"
            )

        # 4. Referencias conservadas para búsqueda
        original_references = {
            ref.upper()
            for ref in REFERENCE_PATTERN.findall(
                f"{doc.title or ''} {doc.body}"
            )
        }

        search_references = {
            ref.upper()
            for ref in REFERENCE_PATTERN.findall(
                search_text
            )
        }

        missing = (
            original_references - search_references
        )

        if missing:
            problems.append(
                f"{path.name}: búsqueda perdió {sorted(missing)}"
            )

        # 5. Contenido no vacío
        if not classification_text:
            problems.append(
                f"{path.name}: clasificación vacía"
            )

        if not search_text:
            problems.append(
                f"{path.name}: búsqueda vacía"
            )

    print(f"TOTAL: {len(documents)} documentos")
    print(f"POR DEPARTAMENTO: {dict(counts)}")
    print(f"PROBLEMAS: {len(problems)}")

    for problem in problems:
        print("ERROR:", problem)

    if problems:
        raise SystemExit(1)

    print("VERIFICACIÓN COMPLETADA")


if __name__ == "__main__":
    main()


