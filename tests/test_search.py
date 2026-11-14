from app.search import (
    DocumentSearchIndex,
    SearchDocument,
)


def create_test_index():
    documents = [
        SearchDocument(
            document_id="DOC-001",
            filename="servidores.txt",
            department="ingenieria",
            title="Problemas del servidor",
            text=(
                "Problemas de latencia en el "
                "servidor de matchmaking."
            ),
        ),

        SearchDocument(
            document_id="DOC-002",
            filename="animaciones.txt",
            department="audiovisual",
            title="Animaciones del personaje",
            text=(
                "Animaciones, texturas y "
                "efectos visuales."
            ),
        ),

        SearchDocument(
            document_id="DOC-003",
            filename="ventas.txt",
            department="negocio",
            title="Reporte comercial",
            text=(
                "Ventas, ingresos y "
                "retención de jugadores."
            ),
        ),
    ]

    return DocumentSearchIndex(documents)


def test_search_returns_relevant_document():
    index = create_test_index()

    results = index.search(
        "problemas de latencia del servidor",
        top_k=3,
    )

    assert results
    assert results[0]["document_id"] == "DOC-001"


def test_search_respects_top_k():
    index = create_test_index()

    results = index.search(
        "animaciones texturas efectos",
        top_k=1,
    )

    assert len(results) <= 1


def test_search_returns_no_duplicates():
    index = create_test_index()

    results = index.search(
        "servidor animaciones ventas",
        top_k=3,
    )

    ids = [
        result["document_id"]
        for result in results
    ]

    assert len(ids) == len(set(ids))


def test_search_empty_query():
    index = create_test_index()

    try:
        index.search("")
        assert False, "Debió rechazar la consulta"
    except ValueError:
        pass