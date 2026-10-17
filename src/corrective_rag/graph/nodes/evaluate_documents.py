from typing import List, Literal
from pydantic import Field, BaseModel
from langchain_core.documents import Document

from ..state import CRAGState



# Quick Note : (For Revision)
# Manual implementation (without LCEL chain):
# We explicitly invoke the prompt template and structured LLM separately.
# First, we create the prompt using prompt.invoke(), then pass it to the
# structured LLM using structured_llm.invoke() to get the evaluation result.
#
# In our current implementation, we use the LCEL chain approach instead.
# The prompt template and structured LLM are connected using the pipe (|)
# operator, so both steps can be executed with a single chain.invoke() call.
#
# Manual implementation:
# structured_llm = llm.with_structured_output(EvaluateDocumentsAnalysis)
# for d in state["documents"]:
#     prompt = docEvaluatePromptTemplate.invoke(
#         {"question": q, "chunk": d.page_content}
#     )
#     output = structured_llm.invoke(prompt)
#
# Chain implementation (used in our project):
# evaluator_llm = llm.with_structured_output(EvaluateDocumentsAnalysis)
# evaluation_chain = document_evaluation_prompt | evaluator_llm
# result = evaluation_chain.invoke(
#     {
#         "question": question,
#         "documentChunk": document.page_content,
#     }
# )


class EvaluateDocumentsAnalysis(BaseModel):
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How relevant the document is to answering the question. "
            "0 means completely irrelevant and 1 means highly relevant."
        ),
    )
    reason: str = Field(description="Short explanation for the relevance score.")


def create_evaluate_documents_node(
    llm,
    document_evaluation_prompt,
    lower_threshold: float,
    upper_threshold: float,
):

    evaluator_llm = llm.with_structured_output(EvaluateDocumentsAnalysis)

    evaluation_chain = document_evaluation_prompt | evaluator_llm

    def evaluate_documents(state: CRAGState):
        """
        This function checks the evaluation of documents based on the state.

        Args:
            state (CRAGState): The current state containing the question and documents.
        Returns:
            dict: Updated state containing good documents,
                  and retrieval verdict.
        """

        question = state["question"]
        documents = state["documents"]

        document_scores: List[float] = []
        good_documents: List[Document] = []

        print("\n========== DOCUMENT EVALUATION ==========")

        for index, document in enumerate(documents):

            result = evaluation_chain.invoke(
                {
                    "question": question,
                    "documentChunk": document.page_content,
                }
            )

            score = result.relevance_score
            document_scores.append(score)

            print(
                f"\nDocument {index + 1}"
                f"\nScore : {score:.2f}"
                f"\nReason: {result.reason}"
            )

            if score >= upper_threshold:
                good_documents.append(document)

        # CORRECT: at least one doc > upper_threshold
        if any(s > upper_threshold for s in document_scores):
            return {
                "good_documents": good_documents,
                "retrieval_verdict": "CORRECT",
                "reason": f"At least one retrieved chunk scored > {upper_threshold}.",
            }

        # INCORRECT: all docs < lower_threshold
        if len(document_scores) > 0 and all(
            s < lower_threshold for s in document_scores
        ):
            return {
                "good_documents": [],
                "retrieval_verdict": "INCORRECT",
                "reason": f"All retrieved chunks scored < {lower_threshold}.",
            }

        # AMBIGUOUS: otherwise
        return {
            "good_documents": good_documents,
            "retrieval_verdict": "AMBIGUOUS",
            "reason": f"No chunk scored > {upper_threshold}, but not all were < {lower_threshold}.",
        }

    return evaluate_documents
