from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
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
# Retriever
#-------------------------

def get_retriever(vector_store):
    retriever = vector_store.as_retriever(
          search_type="similarity",
        search_kwargs={
            "k": 5,
            "fetch_k": 10,
            "lambda_mult": 0.7
        }
    )

    return retriever


#------------------------
# Ask Question
#------------------------

def ask_question(query, vector_store):

    retriever = get_retriever(vector_store)
    docs = retriever.invoke(query)

    context = '\n\n'.join(
      doc.page_content for doc in docs
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