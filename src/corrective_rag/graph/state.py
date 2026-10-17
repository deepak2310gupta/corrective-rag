from typing import Literal, TypedDict
from langchain_core.documents import Document


class CRAGState(TypedDict):

    # Input
    question: str

    # Local retrieval
    documents: list[Document]

    good_documents: list[Document]

    # Retrieval decision
    retrieval_verdict: Literal["CORRECT", "AMBIGUOUS", "INCORRECT"] | None

    # Corrective retrieval
    rewritten_query: str
    web_documents: list[Document]

    refined_context: str

    # Final answer
    answer: str
