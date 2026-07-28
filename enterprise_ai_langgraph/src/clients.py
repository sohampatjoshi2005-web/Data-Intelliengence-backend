from __future__ import annotations

import streamlit as st
import sib_api_v3_sdk
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from presidio_analyzer import AnalyzerEngine
from faster_whisper import WhisperModel

from .config import settings


@st.cache_resource
def get_brevo_client() -> sib_api_v3_sdk.TransactionalEmailsApi:
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key["api-key"] = settings.brevo_api_key
    return sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))


@st.cache_resource
def get_openai_client() -> OpenAI:
    return OpenAI(base_url=settings.api_base_url, api_key=settings.openai_api_key)


@st.cache_resource
def get_analyzer() -> AnalyzerEngine:
    return AnalyzerEngine()


@st.cache_resource
def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
    return chroma_client.get_or_create_collection(name=settings.kb_collection_name, embedding_function=emb_fn)


@st.cache_resource
def get_whisper_model() -> WhisperModel:
    return WhisperModel(settings.whisper_model_name, device="cpu", compute_type="int8")
