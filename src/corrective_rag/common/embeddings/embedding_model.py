from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(embedding_model: str):

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return embeddings
