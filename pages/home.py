import datetime as dt
import os
import random
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import ollama
import pandas as pd
import pandera as pa
import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.llms.ollama import OllamaEndpointNotFoundError
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from requests.exceptions import ConnectionError


load_dotenv()

OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL")
DPATH_VECTORSTORE = Path.cwd() / "data" / "vectorstore"


def add_record(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    n_rows = len(df)
    row_w_meta = (
        {"session_id": st.session_state.session_id}
        | row
        | {"created_at": dt.datetime.now()}
    )
    df.loc[n_rows] = row_w_meta

    return df


def update_record(
    df: pd.DataFrame, record_id: int, updates: dict[str, Any]
) -> pd.DataFrame:
    record = df.loc[record_id]

    # with container:
    #     st.write(updates)
    updated_record = record.to_dict() | updates | {"updated_at": dt.datetime.now()}

    # with container:
    #     st.write(updated_record)
    df.loc[record_id] = updated_record

    return df


def display_message(container, role: str, message: str) -> None:
    with container:
        with st.chat_message(role):
            st.markdown(message)


def display_all_messages(container, df: pd.DataFrame) -> None:
    # for i_row, row in df[:-1].iterrows():
    for i_row, row in df.iterrows():
        content = row["user_input"]
        message = f"**{name}:** {content}"
        display_message(container, role="user", message=message)
        page = row["reference_page"]
        pages = row["pdf_pages"]
        content = row["chatbot_output"]
        pdf_name = row["context_pdf"]
        message = f"**ContextCare:** {content}" + (
            ""
            if page is None
            else f"""

                    See page {page} of {pages} in {pdf_name}
                    """
        )
        display_message(container, role="assistant", message=message)


client = ollama.Client(host=OLLAMA_SERVER_URL)
try:
    models_server = client.list()["models"]
    model_names = [model.model for model in models_server]
except ConnectionError as e:
    st.exception(
        f"Make sure Ollama container is running and `{OLLAMA_SERVER_URL=}` is correctly set."
    )
    st.stop()

schema = pa.DataFrameSchema(
    {
        "session_id": pa.Column(str),
        "user_input": pa.Column(str),
        "chatbot_output": pa.Column(str),
        "response_time_sec": pa.Column(float, pa.Check(lambda x: x > 0)),
        "reference_page": pa.Column(
            "Int64", required=False, nullable=True, coerce=True
        ),
        "context_pdf": pa.Column(str, required=False, nullable=True, coerce=True),
        "pdf_pages": pa.Column("Int64", required=False, nullable=True, coerce=True),
        "user_feedback": pa.Column("Int64", required=False, nullable=True, coerce=True),
        "created_at": pa.Column(pa.DateTime),
        "updated_at": pa.Column(
            pa.DateTime, required=False, nullable=True, coerce=True
        ),
    }
)

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.session_id = str(uuid4())
    st.session_state.df_history = pd.DataFrame(columns=schema.dtypes.keys()).astype(
        {col: str(dtype) for col, dtype in schema.dtypes.items()}
    )
    st.session_state.last_feedback = None
    st.session_state.last_user_message = None
    st.session_state.last_chatbot_message = None

with st.sidebar:
    st.write("# Config")
    name = st.text_input(
        "Name",
        value="User",
        placeholder="Enter your name here! (Defaults to 'User')",
        help="For a more personal experience, enter your name!",
    )
    familiarity = st.selectbox(
        "Article Familiarity Level", ["Novice", "Intermediate", "Expert"]
    )
    model_name = st.selectbox(
        "Base LLM", model_names, help="Main LLM doing the heavy lifting."
    )
    temperature = st.number_input(
        "Temperature",
        min_value=0.0,
        value=0.1,
        help="**Note:** Higher temperature corresponds to respsonses being 'more creative' (less deterministic) but leads hallucinations.",
    )
    embedding_model = st.selectbox(
        "Embedding", model_names, help="Embedding model to encode text."
    )
    options_vectorstores = [v for v in DPATH_VECTORSTORE.glob("*") if v.is_dir()]
    vector_store = st.selectbox(
        "Vector Store",
        options=options_vectorstores,
        format_func=lambda x: x.name,
        help="Vector store for retrieval of context from relevant documents.",
    )

EMBEDDINGS = OllamaEmbeddings(model=embedding_model, base_url=OLLAMA_SERVER_URL)
VECTOR_STORE = FAISS.load_local(
    vector_store, embeddings=EMBEDDINGS, allow_dangerous_deserialization=True
)
LLM = ChatOllama(model=model_name, base_url=OLLAMA_SERVER_URL, temperature=temperature)


st.write("*`Care` with the right **`Context`***")

# === Write out history of messsages
df_history = st.session_state.df_history
st.write(df_history)
container = st.container(height=530)
display_all_messages(container, df_history)

# === Write out new message
updated = None
user_message = st.chat_input("Write your message here!")
if user_message:
    message = f"**{name}:** {user_message}"
    display_message(container, role="user", message=message)
    time_start = dt.datetime.now()
    with st.spinner("Thinking...", show_time=True):
        # TODO: Stream chat so users don't feel like they're waiting
        retriever = VECTOR_STORE.as_retriever()
        prompt_template = f"""You are a health expert and have \
        thoroughly read the study in context. You are tasked with \
        providing a succinct but helpful answer to someone who has a(n) \
        {familiarity} level of familiarity of the study. However, if you don't \
        know the answer, respond with the phrase 'I don't know.'.

        Context: {{context}}
        Question: {{question}}

        Helpful answer:
        """
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain = RetrievalQA.from_chain_type(
            llm=LLM,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        try:
            result = chain.invoke(user_message)
        except OllamaEndpointNotFoundError as e:
            response_content = "**ERROR!** Ollama endpoint is not found. Make sure to pull the model first!"
            st.exception(response_content)
            st.stop()

        top_source = result["source_documents"][0]
        response_content = result["result"]
        page = top_source.metadata.get("page") + 1  # Correct for 0-index
        pages = top_source.metadata.get("total_pages")
        context_pdf = Path(top_source.metadata.get("source")).name
        snippet = top_source.page_content
        time_end = dt.datetime.now()
        display_message(container, role="assistant", message=response_content)

    record = {
        "user_input": user_message,
        "chatbot_output": response_content,
        "response_time_sec": (time_end - time_start).total_seconds(),
        "reference_page": page,
        "pdf_pages": pages,
        "context_pdf": context_pdf,
    }
    updated = add_record(df_history, record)

if len(df_history):
    with container:
        st.write("*Are you satisfied with the response?*")
        feedback = st.feedback(key=len(df_history))
        if feedback != st.session_state.last_feedback:
            updated = update_record(
                df_history, len(df_history) - 1, {"user_feedback": feedback}
            )
            st.session_state.last_feedback = feedback
            st.rerun()

st.session_state.df_history = (
    st.session_state.df_history if updated is None else schema.validate(updated)
)

# === Save session history
# TODO: Write to persistent database
df_history = st.session_state.df_history
if len(df_history) > 0:
    dpath_out = Path.cwd() / "data" / "sessions"
    fname = f"chat_history_session_{st.session_state.session_id}.csv"
    fpath_out = dpath_out / fname
    df_history.to_csv(fpath_out)


# === Restart cache
st.session_state.last_feedback = None

# st.rerun()
