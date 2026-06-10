import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import config
from src.llm import LLMError
from src.rag_pipeline import answer_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class QuestionRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceCitation(BaseModel):
    document_name: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    retrieved_chunks_count: int
    question: str


@router.post("/ask", response_model=ChatResponse)
def ask_question(request: QuestionRequest):
    """Answer a question by retrieving relevant chunks and calling Gemini 2.5 Flash."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    top_k = request.top_k or config.TOP_K

    try:
        result = answer_question(request.question, top_k=top_k)
    except LLMError as exc:
        logger.error("LLM error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error answering question: %s", exc)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceCitation(**s) for s in result["sources"]],
        retrieved_chunks_count=result["retrieved_chunks_count"],
        question=request.question,
    )
