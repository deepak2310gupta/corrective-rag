def create_retriever(vector_store, RETRIEVE_TOP_K):

    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K})
    return retriever