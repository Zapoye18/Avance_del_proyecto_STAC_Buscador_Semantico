from functools import lru_cache
from pathlib import Path

import joblib

from app.extraction import extract_document
from app.preprocessing import prepare_for_classification


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "models/classifier/svm_v1.joblib"
)

MODEL_VERSION = "svm_v1"

DEPARTMENTS = {
    "diseno",
    "ingenieria",
    "audiovisual",
    "calidad",
    "operacion",
    "negocio",
}


@lru_cache(maxsize=1)
def load_classifier():
    """
    Carga el clasificador entrenado.

    Utiliza caché para evitar cargar
    el modelo desde disco en cada consulta.
    """

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


def classify_text(text: str, model=None) -> str:
    """
    Clasifica un texto previamente preparado
    mediante prepare_for_classification().

    El pipeline aplica automáticamente:
      - Limpieza adicional
      - TF-IDF
      - SVM

    Devuelve el departamento predicho.
    """

    if not isinstance(text, str):
        raise TypeError(
            "El texto debe ser una cadena."
        )

    if not text.strip():
        raise ValueError(
            "No se puede clasificar un texto vacío."
        )

    if model is None:
        model = load_classifier()

    prediction = model.predict([text])[0]

    if prediction not in DEPARTMENTS:
        raise ValueError(
            "El modelo devolvió una categoría "
            f"desconocida: {prediction}"
        )

    return str(prediction)


def classify_document(path: str | Path) -> dict:
    """
    Extrae, prepara y clasifica
    un documento nuevo.

    Devuelve un diccionario con
    el resultado de la predicción.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"No existe el documento: {path}"
        )

    # 1. Extraer contenido.
    document = extract_document(path)

    # 2. Preparar el cuerpo sin
    # títulos ni referencias administrativas.
    text = prepare_for_classification(document)

    # 3. Predecir el departamento.
    department = classify_text(text)

    # 4. Construir resultado.
    return {
        "document_id": path.stem,
        "filename": path.name,
        "predicted_department": department,
        "model": MODEL_VERSION,
    }