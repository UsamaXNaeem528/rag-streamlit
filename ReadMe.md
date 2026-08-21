# 📚 RAG Document Assistant

A Hybrid Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its content.

The application combines BM25 lexical retrieval, Chroma semantic search, Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking before generating an answer with Mistral.

## 🌐 Demo

**Live App:**  
https://rag-document-upload.streamlit.app/

**GitHub:**  
https://github.com/UsamaXNaeem528/rag-streamlit

---

## ✨ Features

- 📄 PDF document upload
- ✂️ Document chunking
- 🔤 BM25 lexical retrieval
- 🧠 Chroma vector retrieval
- 🔀 Reciprocal Rank Fusion
- 🎯 BGE Cross-Encoder reranking
- 🤖 Mistral LLM
- 💬 Streamlit chat interface
- 🔒 Session-scoped document isolation
- ☁️ Streamlit deployment

---

## 🏗️ Architecture

```text
PDF
 │
 ▼
Document Loading & Chunking
 │
 ├──────────────┐
 ▼              ▼
BM25          Chroma
 │              │
 └──────┬───────┘
        ▼
    RRF Fusion
        │
        ▼
 Cross-Encoder
    Reranking
        │
        ▼
 Relevant Context
        │
        ▼
    Mistral LLM
        │
        ▼
      Answer
🔎 Retrieval Pipeline
BM25

Provides lexical/keyword-based retrieval and works well with exact terms, technical terminology, and clause numbers.

Chroma

Uses sentence-transformers/all-MiniLM-L6-v2 embeddings for semantic retrieval.

RRF

Combines BM25 and vector retrieval results to improve candidate retrieval.

Cross-Encoder

BAAI/bge-reranker-base reranks the retrieved candidates and selects the most relevant context.

📁 Project Structure
rag-streamlit/
│
├── app.py
├── config.py
├── requirements.txt
│
├── .streamlit/
│   └── config.toml
│
└── rag/
    ├── ingestion.py
    ├── models.py
    ├── storage.py
    ├── retrieval.py
    └── pipeline.py
Modules

app.py — Streamlit UI and application orchestration

ingestion.py — PDF loading, chunking, and cleaning

models.py — Embedding, reranker, and LLM initialization

storage.py — Chroma and BM25

retrieval.py — Hybrid retrieval, RRF, and reranking

pipeline.py — Complete RAG pipeline

⚙️ Setup

Clone the repository:

git clone https://github.com/UsamaXNaeem528/rag-streamlit.git
cd rag-streamlit

Create a virtual environment:

python -m venv .venv

Activate it and install dependencies:

pip install -r requirements.txt

Create .env:

MISTRAL_API_KEY=your_mistral_api_key

Run:

streamlit run app.py
☁️ Deployment

The application is deployed using Streamlit.

The current implementation uses session-scoped storage for uploaded documents and Chroma indexes.

This makes the application suitable for a portfolio/demo environment, but locally stored data should not be considered permanent across application restarts or infrastructure changes.

A production version could move document and vector storage to external persistent services.

🧠 Key Learning

The biggest lesson from this project was that building a RAG pipeline locally and deploying it as an application are two different challenges.

Deployment introduced practical considerations such as:

Dependency management
Model memory
File upload limits
Temporary storage
Session isolation
Cloud resource constraints
🚀 Future Improvements
Persistent vector storage
Persistent document storage
Authentication
Multi-document support
Source/page citations
RAG evaluation
Streaming responses
Document management
Production API backend
🛠️ Tech Stack

Python • LangChain • Streamlit • Chroma • BM25 • Sentence Transformers • Cross-Encoder • Mistral