import logging
import google.generativeai as genai
import config
from src.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

config.validate_config()
genai.configure(api_key=config.GEMINI_API_KEY)

_model = genai.GenerativeModel(config.GEMINI_MODEL)
_generation_config = genai.types.GenerationConfig(
    temperature=0.2,
    max_output_tokens=1024,
    top_p=0.8,
)

_SYSTEM_PROMPT = (
    "You are a precise and helpful document assistant. Your sole purpose is to answer "
    "questions based on the provided document excerpts. Follow these rules strictly:\n\n"
    "1. Base your answer ONLY on the provided context. Do not use outside knowledge.\n"
    "2. If the context does not contain enough information to answer the question, respond with: "
    "\"The uploaded documents do not contain sufficient information to answer this question.\"\n"
    "3. When referencing specific information, mention the page number "
    "(e.g., \"According to page 12, ...\").\n"
    "4. Be concise but complete. Do not pad answers with filler phrases.\n"
    "5. If the question is ambiguous, interpret it in the most reasonable way given the context."
)

_NO_CONTEXT_RESPONSE = (
    "The uploaded documents do not contain sufficient information to answer this question."
)


class LLMError(Exception):
    pass


def generate_answer(question: str, context_chunks: list[RetrievedChunk]) -> str:
    """Build a grounded prompt from retrieved chunks and call Gemini 2.5 Flash."""
    
    print("Retrieved chunks:", len(context_chunks))

    for chunk in context_chunks:
        print("=" * 50)
        print(chunk.document_name)
        print(chunk.page)
        print(chunk.text[:200])

    if not context_chunks:
        return _NO_CONTEXT_RESPONSE

    context_block = "\n\n".join(
        f"--- Excerpt {i + 1} (Source: {chunk.document_name}, Page {chunk.page}) ---\n{chunk.text}"
        for i, chunk in enumerate(context_chunks)
    )

    prompt = (
        f"[SYSTEM]\n{_SYSTEM_PROMPT}\n\n"
        f"[CONTEXT]\n"
        f"The following excerpts were retrieved from the uploaded documents, ordered by relevance:\n\n"
        f"{context_block}\n\n"
        f"[QUESTION]\n{question}\n\n"
        f"[ANSWER]"
    )

    try:
        response = _model.generate_content(
            prompt,
            generation_config=_generation_config
        )
        return response.text
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        raise LLMError(f"Failed to generate answer: {exc}") from exc