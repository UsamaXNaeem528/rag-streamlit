import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_mistralai import ChatMistralAI

from config import settings


@st.cache_resource
def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model
    )


@st.cache_resource
def get_reranker_model():

    return HuggingFaceCrossEncoder(
        model_name=settings.reranker_model
    )


@st.cache_resource
def get_llm():

    return ChatMistralAI(
        model=settings.mistral_model
    )