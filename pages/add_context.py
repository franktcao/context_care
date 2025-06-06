from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama

# from langchain_huggingface import HuggingFaceEmbeddings
VECTORSTORE_PATH = Path("data") / "vectorstore" / "db_pdf_context"
from langchain_ollama import OllamaEmbeddings

model_name = "llama3"
model_name = "tinyllama"
EMBEDDINGS = OllamaEmbeddings(
    model=model_name,
)

CHUNK_SIZE = 1_000
with st.sidebar:
    uploaded = st.file_uploader("Upload document here", type=[".pdf"])

st.write(
    "Use this module to upload a document to use as context for **ContextCare**. To start, upload a PDF on the left."
)

if uploaded:
    temp_saved = "./temp.pdf"
    with open(temp_saved, "wb") as file:
        file.write(uploaded.getvalue())
        file_name = uploaded.name

    loader = PyPDFLoader(temp_saved)
    documents = loader.load()
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=30, separator="\n"
    )

    split = splitter.split_documents(documents)
    content = [s.page_content for s in split]
    # st.write(split)
    # st.write(content)

    # st.write(uploaded)
    # st.write(type(uploaded))
    # pdf_stream = BytesIO(uploaded)
    # pdf_reader = PyPDF2.PdfReader(pdf_stream)

    # loader = PyPDFLoader.PdfFileReader(uploaded)xxx
    # loader = PyPDFLoader(uploaded)
    # st.write(documents)

    # st.write(type(split))

    # load phecode embeddings
    dpath = Path("data") / "dataverse_files"
    fname = "ClinVec_phecode.csv"
    fpath = dpath / fname
    df = pd.read_csv(fpath)

    # get matrix of embeddings
    emb_mat = df.values

    OLLAMA_SERVER_URL = "http://localhost:11434"

    # OLLAMA_SERVER_URL = "http://ollama:11434"
    # OLLAMA_SERVER_URL = "http://ollama-container:11434"
    MODEL_NAME = "tinyllama"
    LLM = Ollama(model=MODEL_NAME, base_url=OLLAMA_SERVER_URL)
    db = FAISS.from_documents(split, EMBEDDINGS)
    db.save_local(VECTORSTORE_PATH)
# db = FAISS.from_documents(split, emb_mat, embedding=)
# db = FAISS.from_embeddings(split, emb_mat)
# db = FAISS.from_embeddings(content, emb_mat)
# db = FAISS.from_embeddings(
#     emb_mat,
#     [f"doc_{i}" for i in range(len(emb_mat))],
# )
# split
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# splitter = CharacterTextSplitter(chunk_size=1_000, chunk_overlap=30, separator="\n")

# split = splitter.split_documents(documents)
# split
