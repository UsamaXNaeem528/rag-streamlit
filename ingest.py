from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


#* Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#* Document Loading
def load_document(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return docs

#* Split Documents
def split_document(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )

    chunks = splitter.split_documents(docs)

    return chunks


#* clean Chunks
def clean_chunks(chunks):

    cleaned_chunks = []

    for chunk in chunks:

        chunk.page_content = str(
            chunk.page_content
        )

        chunk.page_content = (
            chunk.page_content
            .encode(
                "utf-8",
                errors="ignore"
            )
            .decode("utf-8")
        )

        if chunk.page_content.strip():

            cleaned_chunks.append(chunk)

    return cleaned_chunks


#* create vector store
def create_vector_store(chunks, persistent_directory="chroma-db"):

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persistent_directory,

    )

    return vector_store


#* complete ingestion
def ingest_document(file_path, persiste_directory='chroma-db'):

    print('1. LoadPDF')
    docs = load_document(file_path)

    print('2. Splitting / Chunking the Documents')
    chunks = split_document(docs)

    print('3. Clean the documents')
    chunks = clean_chunks(chunks)

    print('4. Create vector store')
    vector_store = create_vector_store(chunks, persiste_directory)

    return vector_store, len(docs), len(chunks)

if __name__ == '__main__':
    file_path = r'C:\Users\Admin\Downloads\iso27001.pdf'
    ingest_document(file_path)
