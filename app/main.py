from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.storage import list_documents, read_document
from app.preprocessing import normalize_text
from app.classification_api import router as classification_router


app = FastAPI(
    title="Norte Lúdico AI",
    version="0.1.0",
    description="Clasificación y búsqueda documental"
)


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


# 2. Registrar el router de clasificación
app.include_router(classification_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def get_documents():
    return {"documents": list_documents()}


@app.get("/documents/{filename}")
def get_document(filename: str):
    try:
        text = read_document(filename)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Documento no encontrado"
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Nombre de archivo inválido"
        )

    return {
        "filename": filename,
        "content": text
    }


@app.post("/preprocess")
def preprocess(request: TextRequest):
    cleaned = normalize_text(request.text)

    return {
        "original": request.text,
        "processed": cleaned
    }
