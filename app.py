import os
import uuid
import tempfile

import streamlit as st

from config import settings

from rag.ingestion import process_pdf

from rag.storage import (
    create_vector_store,
    create_bm25_retriever
)

from rag.models import (
    get_embedding_model,
    get_reranker_model,
    get_llm
)

from rag.pipeline import RAGPipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


# ============================================================
# SESSION STATE
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# MODELS
# ============================================================

embedding_model = get_embedding_model()

reranker_model = get_reranker_model()

llm = get_llm()


# ============================================================
# HEADER
# ============================================================

st.title("📚 RAG Document Assistant")

st.caption(
    "Hybrid RAG: BM25 + Chroma + RRF + "
    "Cross-Encoder Reranking + Mistral"
)


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.subheader("📄 Document")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # File size validation
    # --------------------------------------------------------

    if uploaded_file.size > MAX_FILE_SIZE:

        st.error(
            "File is too large. "
            "Maximum allowed size is 20 MB."
        )

        st.stop()

    # --------------------------------------------------------
    # Process button
    # --------------------------------------------------------

    process_button = st.button(
        "⚡ Process Document",
        type="primary",
        use_container_width=True
    )

    if process_button:

        temp_file_path = None

        try:

            # ------------------------------------------------
            # Create temporary PDF
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                temp_file_path = temp_file.name

            # ------------------------------------------------
            # Process document
            # ------------------------------------------------

            with st.spinner(
                "Processing document..."
            ):

                documents, chunks = process_pdf(
                    temp_file_path
                )

                # --------------------------------------------
                # Generate document ID
                # --------------------------------------------

                document_id = str(
                    uuid.uuid4()
                )

                # --------------------------------------------
                # Storage location
                # --------------------------------------------

                persist_directory = os.path.join(
                    "data",
                    st.session_state.session_id,
                    document_id
                )

                os.makedirs(
                    persist_directory,
                    exist_ok=True
                )

                # --------------------------------------------
                # Create Chroma
                # --------------------------------------------

                vector_store = create_vector_store(
                    chunks=chunks,
                    embedding_model=embedding_model,
                    persist_directory=persist_directory
                )

                # --------------------------------------------
                # Create BM25
                # --------------------------------------------

                bm25_retriever = (
                    create_bm25_retriever(
                        chunks,
                        k=settings.bm25_k
                    )
                )

                # --------------------------------------------
                # Create RAG pipeline
                # --------------------------------------------

                rag_pipeline = RAGPipeline(
                    vector_store=vector_store,
                    bm25_retriever=bm25_retriever,
                    reranker_model=reranker_model,
                    llm=llm
                )

                # --------------------------------------------
                # Store in session
                # --------------------------------------------

                st.session_state.document_id = (
                    document_id
                )

                st.session_state.file_name = (
                    uploaded_file.name
                )

                st.session_state.rag_pipeline = (
                    rag_pipeline
                )

                st.session_state.messages = []

            st.success(
                f"✅ Document processed successfully. "
                f"{len(chunks)} chunks created."
            )

        except Exception as e:

            st.error(
                f"Error processing document: {e}"
            )

        finally:

            # ------------------------------------------------
            # Remove temporary file
            # ------------------------------------------------

            if (
                temp_file_path
                and os.path.exists(temp_file_path)
            ):

                os.remove(
                    temp_file_path
                )


# ============================================================
# ACTIVE DOCUMENT
# ============================================================

if st.session_state.file_name:

    st.info(
        f"📄 Active document: "
        f"**{st.session_state.file_name}**"
    )


# ============================================================
# CHAT
# ============================================================

st.divider()

st.subheader("💬 Ask Questions")


# ------------------------------------------------------------
# Display previous messages
# ------------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ------------------------------------------------------------
# User input
# ------------------------------------------------------------

query = st.chat_input(
    "Ask something about your document..."
)


if query:

    # --------------------------------------------------------
    # Check document
    # --------------------------------------------------------

    if st.session_state.rag_pipeline is None:

        st.warning(
            "Please upload and process "
            "a document first."
        )

        st.stop()

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):

        st.markdown(query)

    # --------------------------------------------------------
    # RAG response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching document and generating answer..."
        ):

            try:

                answer = (
                    st.session_state
                    .rag_pipeline
                    .ask(query)
                )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error(
                    f"Error generating answer: {e}"
                )