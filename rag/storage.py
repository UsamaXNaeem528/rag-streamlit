from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

def create_vector_store(chunks, embedding_model, persist_directory: str):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )


def create_bm25_retriever(chunks, k=5):

    retriever = BM25Retriever.from_documents(chunks)

    retriever.k = k

    return retriever