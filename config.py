import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:

    mistral_model: str = os.getenv(
        "MISTRAL_MODEL",
        "mistral-small-2506"
    )

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-reranker-base"
    )

    reranker_model: str = os.getenv(
        "RERANKER_MODEL",
        "BAAI/bge-reranker-base"
    )

    chunk_size: int = int(
        os.getenv("CHUNK_SIZE", "1000")
    )

    chunk_overlap: int = int(
        os.getenv("CHUNK_OVERLAP", "100")
    )

    vector_k: int = int(
        os.getenv("VECTOR_K", "5")
    )

    bm25_k: int = int(
        os.getenv("BM25_K", "5")
    )

    rerank_top_n: int = int(
        os.getenv("RERANK_TOP_N", "10")
    )


settings = Settings()