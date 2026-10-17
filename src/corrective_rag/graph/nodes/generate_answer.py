from ..state import CRAGState


def create_generate_answer_node(
    llm,
    generation_prompt,
):

    generation_chain = generation_prompt | llm

    def generate_answer(state: CRAGState):
        """
        This function generates an answer based on the state.

        Args:
            state (CRAGState): The current state containing the question and documents.
        Returns:
            dict: A dictionary containing the generated answer.
        """

        question = state["question"]
        context = state["refined_context"]

        print("\n========== GENERATION ==========")

        result = generation_chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )

        answer = result.content

        return {
            "answer": answer,
        }

    return generate_answer
