from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader


def load_documents(directory_path: str):

    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    return documents