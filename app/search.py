from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.extraction import extract_document
from app.preprocessing import prepare_for_search


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/corpus"


@dataclass
class SearchDocument:
    document_id: str
    filename: str
    department: str
    title: str
    text: str


class DocumentSearchIndex:
    """
    Índice documental basado en TF-IDF.

    Cada documento ocupa una fila
    en la matriz de características.
    """

    def __init__(self, documents):
        if not documents:
            raise ValueError(
                "No hay documentos para indexar."
            )

        identifiers = [
            doc.document_id
            for doc in documents
        ]

        if len(identifiers) != len(set(identifiers)):
            raise ValueError(
                "Existen identificadores duplicados."
            )

        self.documents = documents

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            norm="l2",
        )

        texts = [
            doc.text
            for doc in documents
        ]

        self.matrix = self.vectorizer.fit_transform(
            texts
        )

    def search(self, query: str, top_k: int = 5):
        """
        Devuelve los documentos más relevantes
        ordenados por similitud coseno.
        """

        if not isinstance(query, str):
            raise TypeError(
                "La consulta debe ser texto."
            )

        if not query.strip():
            raise ValueError(
                "La consulta no puede estar vacía."
            )

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError(
                "top_k debe ser un número entero."
            )

        if top_k < 1:
            raise ValueError(
                "top_k debe ser mayor que cero."
            )

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.matrix,
        )[0]

        ranking = np.argsort(
            -similarities,
            kind="stable",
        )

        results = []

        for index in ranking:
            score = float(similarities[index])

            # No devolvemos documentos sin
            # coincidencia con el vocabulario.
            if score <= 0:
                continue

            document = self.documents[index]

            results.append({
                "document_id": document.document_id,
                "filename": document.filename,
                "department": document.department,
                "title": document.title,
                "score": round(score, 4),
            })

            if len(results) >= top_k:
                break

        return results


def build_search_index(corpus=CORPUS):
    """
    Extrae los documentos del corpus
    y construye el índice TF-IDF.
    """

    corpus = Path(corpus)

    documents = []

    for path in sorted(corpus.rglob("*.txt")):
        extracted = extract_document(path)

        text = prepare_for_search(
            extracted
        )

        if not text.strip():
            continue

        documents.append(
            SearchDocument(
                document_id=path.stem,
                filename=path.name,
                department=path.parent.name,
                title=extracted.title or path.stem,
                text=text,
            )
        )

    if not documents:
        raise ValueError(
            f"No se encontraron documentos en {corpus}"
        )

    return DocumentSearchIndex(
        documents
    )