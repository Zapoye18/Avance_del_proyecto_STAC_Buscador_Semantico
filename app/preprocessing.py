import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normaliza texto sin eliminar información semántica."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()