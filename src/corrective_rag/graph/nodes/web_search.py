from langchain_core.documents import Document

from ..state import CRAGState


def create_web_search_node(web_search):

    def web_search_node(state: CRAGState):

        query = state["rewritten_query"]

        print("\n========== CORRECTIVE WEB SEARCH ==========")
        print(f"Search query: {query}")

        results = web_search.invoke(query)

        web_documents = []

        for result in results:

            if isinstance(result, Document):
                web_documents.append(result)

            else:
                web_documents.append(
                    Document(
                        page_content=str(result)
                    )
                )

        print(f"Web documents retrieved: " f"{len(web_documents)}")

        return {
            "web_documents": web_documents,
        }

    return web_search_node
