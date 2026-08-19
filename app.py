import os
import tempfile

import streamlit as st

from ingest import ingest_document
from main import ask_question


# ==================================================
# Page configuration
# ==================================================

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# Session state
# ==================================================

if "vector_store" not in st.session_state:

    st.session_state.vector_store = None


if "file_name" not in st.session_state:

    st.session_state.file_name = None


if "messages" not in st.session_state:

    st.session_state.messages = []


# ==================================================
# Title
# ==================================================

st.title("📚 RAG Document Assistant")

st.write(
    "Upload a PDF and ask questions about its content."
)


# ==================================================
# Sidebar
# ==================================================

with st.sidebar:

    st.header("📄 Upload Document")


    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )


    if uploaded_file is not None:

        st.info(
            f"Selected: {uploaded_file.name}"
        )


        process_button = st.button(
            "Process Document",
            type="primary",
            use_container_width=True
        )


        if process_button:

            # ------------------------------------------
            # Create temporary PDF
            # ------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                temp_file_path = temp_file.name


            try:

                # --------------------------------------
                # Process document
                # --------------------------------------

                with st.spinner(
                    "Reading and processing document..."
                ):

                    (
                        vector_store,
                        page_count,
                        chunk_count
                    ) = ingest_document(
                        temp_file_path
                    )


                # --------------------------------------
                # Save in session state
                # --------------------------------------

                st.session_state.vector_store = (
                    vector_store
                )

                st.session_state.file_name = (
                    uploaded_file.name
                )

                # Clear old chat

                st.session_state.messages = []


                # --------------------------------------
                # Success
                # --------------------------------------

                st.success(
                    "Document processed successfully!"
                )

                st.write(
                    f"📄 Pages: {page_count}"
                )

                st.write(
                    f"🧩 Chunks: {chunk_count}"
                )


            except Exception as e:

                st.error(
                    f"Error processing document: {e}"
                )


            finally:

                # --------------------------------------
                # Delete temporary PDF
                # --------------------------------------

                if os.path.exists(
                    temp_file_path
                ):

                    os.remove(
                        temp_file_path
                    )


    # ----------------------------------------------
    # Current document
    # ----------------------------------------------

    if st.session_state.file_name:

        st.divider()

        st.subheader("Current document")

        st.write(
            f"📄 {st.session_state.file_name}"
        )


# ==================================================
# Chat history
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ==================================================
# Chat input
# ==================================================

query = st.chat_input(
    "Ask something about your document..."
)


if query:

    # ----------------------------------------------
    # Make sure document exists
    # ----------------------------------------------

    if st.session_state.vector_store is None:

        st.warning(
            "Please upload and process a document first."
        )

        st.stop()


    # ----------------------------------------------
    # User message
    # ----------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })


    with st.chat_message("user"):

        st.markdown(query)


    # ----------------------------------------------
    # AI response
    # ----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the document..."
        ):

            try:

                answer = ask_question(
                    query,
                    st.session_state.vector_store
                )


                st.markdown(answer)


                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })


            except Exception as e:

                st.error(
                    f"Error: {e}"
                )