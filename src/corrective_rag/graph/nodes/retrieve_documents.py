from ..state import CRAGState


def create_retrieve_documents_node(retriever):

    def retrieve_documents(state: CRAGState):
        """
        This function retrieves documents based on the question in the state.

        Args:
            state (CRAGState): The current state containing the question.
        Returns:
            list: A list of retrieved documents.

        """

        question = state["question"]

        print("\n========== LOCAL RETRIEVAL ==========")
        print(f"Query: {question}")

        documents = retriever.invoke(question)

        print(f"Retrieved documents: {len(documents)}")

        return {
            "documents": documents,
        }

    return retrieve_documents
