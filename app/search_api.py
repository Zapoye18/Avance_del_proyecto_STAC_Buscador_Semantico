from functools import lru_cache

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.search import build_search_index


router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Consulta en lenguaje natural",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Número máximo de documentos",
    )


class SearchResult(BaseModel):
    document_id: str
    filename: str
    department: str
    title: str
    score: float


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[SearchResult]


@lru_cache(maxsize=1)
def get_search_index():
    """
    Construye el índice una vez
    y lo reutiliza entre consultas.
    """

    return build_search_index()


@router.post(
    "/search",
    response_model=SearchResponse,
)
def search_documents(request: SearchRequest):
    """
    Busca documentos utilizando
    TF-IDF y similitud coseno.
    """

    results = get_search_index().search(
        query=request.query,
        top_k=request.top_k,
    )

    return SearchResponse(
        query=request.query,
        total_results=len(results),
        results=results,
    )