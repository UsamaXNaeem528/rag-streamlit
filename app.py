import os
import tempfile

import streamlit as st

# Note: Ensure ingest.py exports 'ingest_document' or 'complete_ingestion'
from ingest import ingest_document
from main import ask_question


# ==================================================
# Page configuration
# ==================================================

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
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
# Header
# ==================================================

st.title("📚 RAG Document Assistant")
st.caption("Upload a PDF to start asking questions about its content.")

# ==================================================
# Main Layout: Document Uploader (Mobile & Desktop Friendly)
# ==================================================

# Keep upload section open by default, close it once a file is active
expander_default = st.session_state.vector_store is None

with st.expander("📄 Document Management", expanded=expander_default):
    
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            label_visibility="visible"
        )

    with col2:
        st.write("") # Spacing for vertical alignment
        st.write("")
        process_button = st.button(
            "⚡ Process Document",
            type="primary",
            use_container_width=True,
            disabled=(uploaded_file is None)
        )

    # Show active document status inside the card
    if st.session_state.file_name:
        st.success(f"**Active Document:** {st.session_state.file_name}")

    if process_button and uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        try:
            with st.spinner("Reading and processing document..."):
                (
                    vector_store,
                    page_count,
                    chunk_count
                ) = ingest_document(temp_file_path)

            st.session_state.vector_store = vector_store
            st.session_state.file_name = uploaded_file.name
            st.session_state.messages = []  # Clear old chat history

            st.toast("Document processed successfully!", icon="✅")
            st.rerun()

        except Exception as e:
            st.error(f"Error processing document: {e}")

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

st.divider()

# ==================================================
# Chat Interface
# ==================================================

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query Input
query = st.chat_input("Ask something about your document...")

if query:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process a document first.")
        st.stop()

    # Append user input
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            try:
                answer = ask_question(query, st.session_state.vector_store)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating answer: {e}")