import argparse

from app.search import build_search_index


def main():
    parser = argparse.ArgumentParser(
        description="Buscador documental TF-IDF"
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Consulta en lenguaje natural.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Cantidad máxima de resultados.",
    )

    args = parser.parse_args()

    index = build_search_index()

    results = index.search(
        query=args.query,
        top_k=args.top_k,
    )

    print("\n=== RESULTADOS DE BÚSQUEDA ===")
    print(f"Consulta: {args.query}")
    print(f"Resultados: {len(results)}")

    for position, result in enumerate(
        results,
        start=1,
    ):
        print("\n" + "-" * 60)

        print(f"Posición: {position}")
        print(f"Documento: {result['filename']}")
        print(f"Departamento: {result['department']}")
        print(f"Título: {result['title']}")
        print(f"Similitud: {result['score']:.4f}")


if __name__ == "__main__":
    main()