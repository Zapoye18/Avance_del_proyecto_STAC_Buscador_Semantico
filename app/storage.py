from pathlib import Path

DATA_DIR = Path("data/documents")


def list_documents() -> list[str]:
    """Devuelvera los nombres de documentos txt. enlistados en el directorio de datos."""
    return sorted(
        file.name
        for file in DATA_DIR.glob("*.txt")
        if file.is_file()
    )


def read_document(filename: str) -> str:
    """Lee un documento de texto de la carpeta de datos."""
    if (
        not filename
        or Path(filename).name != filename
        or not filename.endswith(".txt")
    ):
        raise ValueError("Nombre de archivo inválido")

    path = DATA_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(filename)

    return path.read_text(encoding="utf-8")