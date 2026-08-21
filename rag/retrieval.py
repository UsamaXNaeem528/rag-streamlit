from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker
)


def create_hybrid_retriever(
    vector_store,
    bm25_retriever,
    reranker_model,
    vector_k=5,
    rerank_top_n=5
):

    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": vector_k
        }
    )

    hybrid = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_retriever
        ],
        weights=[
            0.4,
            0.6
        ],
        c=60
    )

    reranker = CrossEncoderReranker(
        model=reranker_model,
        top_n=rerank_top_n
    )

    return ContextualCompressionRetriever(
        base_retriever=hybrid,
        base_compressor=reranker
    )