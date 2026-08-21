from langchain_core.prompts import ChatPromptTemplate

from rag.retrieval import create_hybrid_retriever


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful document question-answering assistant.

Answer the user's question ONLY using the provided context.

If the answer cannot be found in the context, say:

"I could not find the answer in the document."

Do not invent information.
"""
    ),
    (
        "human",
        """
Context:

{context}

Question:

{question}
"""
    )
])


class RAGPipeline:

    def __init__(self, vector_store, bm25_retriever, reranker_model, llm):
        self.llm = llm
        self.retriever = create_hybrid_retriever(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            reranker_model=reranker_model
        )

    def ask(self, question: str):

        documents = self.retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        prompt = PROMPT.invoke({
            "context": context,
            "question": question
        })

        response = self.llm.invoke(prompt)

        return response.content