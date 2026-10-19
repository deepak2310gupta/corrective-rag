from langgraph.graph import StateGraph, START, END

from .nodes.retrieve_documents import create_retrieve_documents_node
from .nodes.evaluate_documents import create_evaluate_documents_node
from .nodes.generate_answer import create_generate_answer_node
from .nodes.rewrite_query import create_rewrite_query_node
from .nodes.web_search import create_web_search_node
from .nodes.refine import create_refine_node

from .routing import route_after_evaluate_documents

from .state import CRAGState


def build_crag_graph(
    retriever,
    llm,
    document_evaluation_prompt,
    generation_prompt,
    rewrite_prompt,
    lower_threshold,
    upper_threshold,
    web_search,
):

    graph = StateGraph(CRAGState)

    # Defining the nodes with their respective dependencies
    retrieve_documents_node = create_retrieve_documents_node(retriever)
    evaluate_documents_node = create_evaluate_documents_node(
        llm, document_evaluation_prompt, lower_threshold, upper_threshold
    )
    generate_answer_node = create_generate_answer_node(llm, generation_prompt)
    rewrite_query_node = create_rewrite_query_node(llm, rewrite_prompt)
    web_search_node = create_web_search_node(web_search)
    refine_node = create_refine_node()

    # Adding nodes to the graph
    graph.add_node("retrieve_documents", retrieve_documents_node)
    graph.add_node("evaluate_documents", evaluate_documents_node)
    graph.add_node("refine", refine_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("web_search", web_search_node)

    # Adding edges to define the workflow
    graph.add_edge(START, "retrieve_documents")
    graph.add_edge("retrieve_documents", "evaluate_documents")

    graph.add_conditional_edges(
        "evaluate_documents",
        route_after_evaluate_documents,
        {
            "RELEVANT": "refine",
            "NOT_RELEVANT": "rewrite_query",
        },
    )

    graph.add_edge("rewrite_query", "web_search")
    graph.add_edge("web_search", "refine")

    graph.add_edge("refine", "generate_answer")
    graph.add_edge("generate_answer", END)

    # Compile the graph
    return graph.compile()
