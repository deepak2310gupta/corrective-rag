from ..state import CRAGState


def create_refine_node():
    def refine(state: CRAGState):

        if state["retrieval_verdict"] == "CORRECT":
            refined_context = "\n\n".join([d.page_content for d in state["good_documents"]])
        elif state["retrieval_verdict"] == "AMBIGUOUS":
            refined_context = "\n\n".join(
                [d.page_content for d in state["good_documents"] + state["web_documents"]]
            )
        else:
            refined_context = "\n\n".join([d.page_content for d in state["web_documents"]])

        return {"refined_context": refined_context}

    return refine
