import datetime
from io import BytesIO
import os
from pathlib import Path
import time

import ollama
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel

# from langchain_huggingface import HuggingFaceEmbeddings


CHUNK_SIZE = 1024
CHUNK_OVERLAP = 32
DPATH_VECTORSTORE = Path.cwd() / "data" / "vectorstore"
VECTORSTORE_DPATH = Path("data") / "vectorstore"

load_dotenv()
OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL")


class VectorStoreConfig(BaseModel):
    # model: OllamaEmbeddings
    chunk_size: int
    chunk_overlap: int
    fpath_pdf: str | Path
    fpath_vector_store: str | Path


def hash_pydantic_model(model: BaseModel):
    return hash(model.model_dump_json())


def hash_ollama_model(model: OllamaEmbeddings):
    return model.model


@st.cache_data(
    hash_funcs={
        VectorStoreConfig: hash_pydantic_model,
        OllamaEmbeddings: hash_ollama_model,
    }
)
def run_and_save(embedding_model, config: VectorStoreConfig) -> str:
    with st.spinner("Running...", show_time=True):
        fpath_pdf = config.fpath_pdf
        docs = load_documents(fpath_pdf)

        chunk_size = config.chunk_size
        chunk_overlap = config.chunk_overlap
        split = split_docs(docs, chunk_size, chunk_overlap)

        # embedding_model = config.model
        vector_store = construct_vector_store(split, embedding_model)

    with st.spinner("Saving...", show_time=True):
        fpath_vector_store = config.fpath_vector_store
        fpath_saved = save_vector_store(vector_store, fpath_vector_store)

    st.write(f"**Success**! Vector store has been saved to `{fpath_vector_store}`")

    return fpath_saved


@st.cache_data
def load_documents(fpath: str | Path):
    loader = PyPDFLoader(fpath)
    documents = loader.load()

    return documents


@st.cache_data
def split_docs(_documents, chunk_size: int, chunk_overlap: int):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n"
    )
    with st.spinner("Splitting documents...", show_time=True):
        split = splitter.split_documents(_documents)

    return split


def construct_vector_store(_split_docs, embedding_model: OllamaEmbeddings):
    with st.spinner("Constructing vector store...", show_time=True):
        vector_store = FAISS.from_documents(_split_docs, embedding_model)
    return vector_store


@st.cache_data
def save_vector_store(_vector_store, fpath: str) -> None:
    with st.spinner("Saving vector store locally...", show_time=True):
        _vector_store.save_local(fpath)

    return fpath


@st.cache_data
def save_pdf_locally(uploaded) -> str:
    fpath_temp = "./temp.pdf"
    with open(fpath_temp, "wb") as file:
        file.write(uploaded.getvalue())

    return fpath_temp


with st.sidebar:
    st.write("# Upload")
    uploaded = st.file_uploader("Upload document here", type=[".pdf"])

client = ollama.Client(host=OLLAMA_SERVER_URL)
try:
    models_server = client.list()["models"]
    model_names = [model.model for model in models_server]
except ConnectionError as e:
    st.exception(
        f"Make sure Ollama container is running and `{OLLAMA_SERVER_URL=}` is correctly set."
    )
    st.stop()

if not uploaded:
    st.write(
        "Use this module to upload a document to use as context for **ContextCare**. To start, upload a PDF on the left."
    )
else:
    st.write("Select embedding model and name for vector store.")
    fname = Path(uploaded.name)
    embedding_model = st.selectbox(
        "Select Embedding Model", model_names, help="Embedding to encode text."
    )
    options_vectorstores = [v for v in DPATH_VECTORSTORE.glob("*") if v.is_dir()]
    default_name = f"vs__{embedding_model.replace(':latest', '')}__{fname.stem}"
    vector_store_name = st.text_input(
        "Save vector store as...",
        default_name,
        help="Vector store for retrieval will be saved in `data/vectorstore/`.",
    )
    EMBEDDINGS = OllamaEmbeddings(model=embedding_model, base_url=OLLAMA_SERVER_URL)
    fpath_temp = save_pdf_locally(uploaded)

    # TODO: Add database connection for vector store
    st.write("When ready, run and save the vector store.")

    fpath_vector_store = VECTORSTORE_DPATH / vector_store_name
    config = VectorStoreConfig(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        fpath_pdf=fpath_temp,
        fpath_vector_store=fpath_vector_store,
    )
    # st.write(config.model_dump_json())
    # st.write(EMBEDDINGS)
    run = st.button("Submit", on_click=run_and_save, args=(EMBEDDINGS, config))

    # st.stop()
