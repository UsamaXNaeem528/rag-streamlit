# main.py
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import bm25_retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from dotenv import load_dotenv
load_dotenv()

#--------------------------
# LLM, embedding model
#--------------------------

llm = ChatMistralAI(
    model = 'mistral-small-2506'
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory='chroma-db',
    embedding_function=embedding_model
)

#--------------------------
# Prompt Template
#--------------------------

template = ChatPromptTemplate.from_messages([
    ("system",
      """
        You are a helpful AI assistant. User only the provided context to answer the question.
        If the answer is not present in the context,
        say: "I could not find the answer in the document.
    """),
    
    ("human", '''
    Context:
    {context}

    Question:
    {question}
    ''')
])


#-------------------------
# Retriever Hyrbid(Vector + B2)
#-------------------------

def vector_retriever(vector_store):
    vector_retriever = vector_store.as_retriever(
          search_type="similarity",
        search_kwargs={
            "k": 5,
        }
    )
    return vector_retriever


def hybrid_retriever(vector_store):

    '''RRF (Reciprocal Rank Fusion) combines document lists by their ranks.'''
    bm25_vector_retriver = EnsembleRetriever(
        retrievers = [bm25_retriever(), vector_retriever(vector_store)],
        weights = [0.4, 0.6],    #0.4 = bm25m , 0.6 = vector retriever
        c=60  #RRF constant
        )

    '''Cross Encoder Reranker'''
    cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    reranker = CrossEncoderReranker(
        model = cross_encoder_model,
        top_n=10
    )

    '''CONTEXTUAL COMPRESSION RETRIEVER (Hybrid Retriever -> Cross Encoder -> Top n chunks)'''
    compression_retriever = ContextualCompressionRetriever(
        base_retriever = bm25_vector_retriver,
        base_compressor = reranker
    )

    return compression_retriever


#------------------------
# Ask Question
#------------------------

def ask_question(query, vector_store):

    retriever = hybrid_retriever(vector_store)
    relevant_docs = retriever.invoke(query)

    context = '\n\n'.join(
      doc.page_content for doc in relevant_docs
    )

    final_prompt = template.invoke({
        'context' : context,
        'question' : query
    })

    response = llm.invoke(final_prompt)

    return response.content

if __name__ == '__main__':
    query = 'What does Clause 5.2 require regarding the Information Security Policy?'
    response = ask_question(query, vector_store)
    print(response)