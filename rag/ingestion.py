from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str):

    loader = PyPDFLoader(file_path)

    return loader.load()


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    return splitter.split_documents(documents)


def clean_documents(documents):

    cleaned = []

    for doc in documents:

        text = str(doc.page_content)

        text = (
            text
            .encode("utf-8", errors="ignore")
            .decode("utf-8")
            .strip()
        )

        if not text:
            continue

        doc.page_content = text

        cleaned.append(doc)

    return cleaned


def process_pdf(file_path: str):

    documents = load_pdf(file_path)

    chunks = split_documents(documents)

    chunks = clean_documents(chunks)

    return documents, chunks