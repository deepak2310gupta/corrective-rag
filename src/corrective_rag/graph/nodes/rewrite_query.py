from pydantic import Field, BaseModel

from ..state import CRAGState


# Quick Note (Revision):
# Manual implementation (without LCEL chain):
# First, invoke the prompt template separately to generate the prompt
# using the original question.
# Then, pass the generated prompt to the structured LLM using .invoke()
# to get the rewritten query in the QueryRewrite Pydantic format.
#
# In our current implementation, we use an LCEL chain instead.
# The rewrite prompt and structured LLM are connected using the pipe (|)
# operator, allowing us to execute both steps with a single .invoke() call.
#
# Manual implementation:
# prompt = rewriteQueryPromptTemplate.invoke(
#     {
#         "question": state["question"],
#     }
# )
#
# structured_llm = llm.with_structured_output(QueryRewrite)
# response = structured_llm.invoke(prompt)
#
# Chain implementation (used in our project):
# structured_llm = llm.with_structured_output(QueryRewrite)
# rewrite_chain = rewrite_prompt | structured_llm
# response = rewrite_chain.invoke({"question": question})


class QueryRewrite(BaseModel):
    restructured_query: str = Field(description="Improved query for web search.")


def create_rewrite_query_node(llm, rewrite_prompt):

    structured_llm = llm.with_structured_output(QueryRewrite)

    rewrite_chain = rewrite_prompt | structured_llm

    def rewrite_query(state: CRAGState):
        """
        This function rewrites the query based on the state.

        Args:
            state (CRAGState): The current state containing the question and documents.
        Returns:
            dict: A dictionary containing the rewritten query.
        """

        question = state["question"]

        print("\n========== QUERY REWRITING ==========")
        print(f"Original query: {question}")

        response = rewrite_chain.invoke({"question": question})

        rewritten_query = response.restructured_query

        print(f"Rewritten query: {rewritten_query}")

        return {
            "rewritten_query": rewritten_query,
        }

    return rewrite_query

