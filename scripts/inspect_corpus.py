"""Validate corpus extraction without modifying original documents."""
from collections import Counter
from pathlib import Path
from app.extraction import extract_document
from app.preprocessing import prepare_for_classification, prepare_for_search

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"


def main():
    docs = sorted(CORPUS.rglob("*.txt"))
    if not docs:
        raise SystemExit(f"No hay archivos TXT en {CORPUS}")
    stats = Counter()
    for file in docs:
        doc = extract_document(file)
        prepare_for_classification(doc)
        prepare_for_search(doc)
        stats[file.parent.name] += 1
        print(f"OK {file.relative_to(CORPUS)} | rol={doc.author_role or 'NO DISPONIBLE'} | titulo={bool(doc.title)} | avisos={len(doc.extraction_warnings)}")
    print("TOTAL:", len(docs), "DOCUMENTOS")
    print("POR CARPETA:", dict(sorted(stats.items())))


if __name__ == "__main__":
    main()