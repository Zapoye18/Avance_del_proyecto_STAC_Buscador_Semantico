import tempfile

from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from pydantic import BaseModel

from app.extraction import extract_document
from app.preprocessing import prepare_for_classification
from app.classification import (
    classify_text,
    MODEL_VERSION,
)


router = APIRouter(tags=["classification"])

MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB


class ClassificationResponse(BaseModel):
    document_id: str
    filename: str
    predicted_department: str
    model: str


@router.post(
    "/classify",
    response_model=ClassificationResponse,
)
def classify_uploaded_document(
    file: UploadFile = File(...),
):
    """
    Recibe un documento TXT y predice
    el departamento al que pertenece.

    No guarda el documento en el corpus.
    """

    filename = Path(file.filename or "").name

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="El archivo no tiene nombre.",
        )

    if Path(filename).suffix.lower() != ".txt":
        raise HTTPException(
            status_code=415,
            detail=(
                "Por ahora solamente "
                "se permiten archivos TXT."
            ),
        )

    contents = file.file.read(
        MAX_FILE_SIZE + 1
    )

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="El archivo está vacío.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="El archivo supera los 2 MB.",
        )

    temporary_path = None

    try:
        # Crear un archivo temporal.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".txt",
            delete=False,
        ) as temporary:
            temporary.write(contents)
            temporary_path = Path(temporary.name)

        # Extraer el documento.
        document = extract_document(
            temporary_path
        )

        # Preparación base.
        text = prepare_for_classification(
            document
        )

        # El pipeline realiza su limpieza
        # adicional, TF-IDF y predicción.
        department = classify_text(text)

        return ClassificationResponse(
            document_id=Path(filename).stem,
            filename=filename,
            predicted_department=department,
            model=MODEL_VERSION,
        )

    except UnicodeError:
        raise HTTPException(
            status_code=422,
            detail=(
                "No se pudo interpretar "
                "el contenido del archivo."
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    finally:
        file.file.close()

        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )