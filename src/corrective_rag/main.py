from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

from .graph.workflow import build_crag_graph

from .common.ingestion.loader import load_documents
from .common.chunking.splitter import split_documents
from .common.embeddings.embedding_model import get_embedding_model
from .common.vectorStore.vector_store import create_vector_store
from .common.retriever.retriever import create_retriever
from .common.generation.llm import create_llm
from .common.generation.prompt import (
    create_answer_prompt,
    create_rewrite_query_prompt,
    create_doc_eval_prompt,
)


def main():

    load_dotenv()

    # Infrastructure
    retrieve_topK = 5
    DIRECTORY_PATH = "src/corrective_rag/data"
    documents = load_documents(DIRECTORY_PATH)
    chunks = split_documents(documents)
    embeddings = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    vector_store = create_vector_store(chunks, embeddings)
    retriever = create_retriever(vector_store, retrieve_topK)
    llm = create_llm()

    # Web search
    web_search = DuckDuckGoSearchRun()

    # Prompts
    document_evaluation_prompt = create_doc_eval_prompt()
    generation_prompt = create_answer_prompt()
    rewrite_query_prompt = create_rewrite_query_prompt()

    # Build Graph
    graph = build_crag_graph(
        retriever=retriever,
        llm=llm,
        document_evaluation_prompt=document_evaluation_prompt,
        generation_prompt=generation_prompt,
        rewrite_prompt=rewrite_query_prompt,
        lower_threshold=0.30,
        upper_threshold=0.70,
        web_search=web_search,
    )

    question = input("\nAsk a question: ")

    result = graph.invoke(
        {
            "question": question,
            "documents": [],
            "good_documents": [],
            "retrieval_verdict": None,
            "web_documents": [],
            "rewritten_query": "",
            "refined_context": "",
            "answer": "",
        }
    )

    print("\n========================================")
    print("*** FINAL ANSWER ***")
    print("==========================================")
    print(result["answer"])


if __name__ == "__main__":
    main()
