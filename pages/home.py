import os
import logging
from pathlib import Path
import traceback
import requests
import json

import ollama
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.llms.ollama import OllamaEndpointNotFoundError
from langchain_community.vectorstores import FAISS
from requests.exceptions import ConnectionError
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain import PromptTemplate
from dotenv import load_dotenv

# OLLAMA_SERVER_URL = "http://0.0.0.0:11434"
# OLLAMA_SERVER_URL = "http://localhost:11434"
# OLLAMA_SERVER_URL = "http://ollama:11434"
# OLLAMA_SERVER_URL = "http://ollama-container:11434"
# MODEL_NAME = "tinyllama"
# MODEL_NAME = "deepseek-r1"

load_dotenv()
OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL")
DPATH_VECTORSTORE = Path.cwd() / "data" / "vectorstore"

client = ollama.Client(host=OLLAMA_SERVER_URL)
models = client.list()

model = st.selectbox("Models", models, format_func=lambda x: x.model)
MODEL_NAME = model.model

with st.sidebar:
    st.write("# Config")
    name = st.text_input(
        "Name",
        placeholder="Enter your name here! (Defaults to 'User'",
        help="For a more personal experience, enter your name!",
    )
    familiarity = st.selectbox(
        "Article Familiarity Level", ["Novice", "Intermediate", "Expert"]
    )
    model_name = st.selectbox(
        "Base LLM", [MODEL_NAME], help="Main LLM doing the heavy lifting."
    )
    temperature = st.number_input(
        "Temperature",
        min_value=0.0,
        value=0.1,
        help="**Note:** Higher temperature corresponds to respsonses being 'more creative' (less deterministic) but leads hallucinations.",
    )
    embedding_model = st.selectbox(
        "Embedding", [MODEL_NAME], help="Embedding model to encode text."
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
retriever = VECTOR_STORE.as_retriever()
# LLM = Ollama(model=MODEL_NAME, base_url=OLLAMA_SERVER_URL)
LLM = ChatOllama(model=model_name, base_url=OLLAMA_SERVER_URL, temperature=temperature)
# response_content = LLM.invoke("why is the sky blue?")
# st.write(response_content)
st.write("*`Care` with the right **`Context`***")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.message_id = 0
if name == "":
    name = "User"

container = st.container(height=530)
for i_message, message in enumerate(st.session_state.messages):
    with container:
        role = message["role"]
        display_name = "ContextCare" if role == "assistant" else name
        with st.chat_message(role):
            content = message["content"]
            display_message = f"**{display_name}:** {content}"
            st.markdown(display_message)
            if role == "assistant":
                st.write("*Did you find this response helpful?*")
                feedback = st.feedback("faces", key=i_message)

user_message = st.chat_input("Write your message here!")
if user_message:
    role = "user"
    st.session_state.messages.append({"role": role, "content": user_message})
    with container:
        with st.chat_message(role):
            display_name = "ContextCare" if role == "assistant" else name
            display_message = f"**{display_name}:** {user_message}"
            st.markdown(display_message)
        with st.spinner("Thinking...", show_time=True):
            # TODO: Stream chat so users don't feel like they're waiting
            retriever = VECTOR_STORE.as_retriever()
            prompt_template = f"""You are a health expert and have \
            thoroughly read the study in context. You are tasked with \
            providing a helpful answer to a question that someone who has
            a(n) {familiarity} level of familiarity of the study. However, \
            if you don't know the answer, just say that you don't know. Please
            be succinct.

            Context: {{context}}
            Question: {{question}}

            Only return answers that are helpful below and nothing else.
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
            # response_content = LLM.invoke(user_message)
            # response_content = LLM.invoke(user_message).content

            try:
                result = chain.invoke(user_message)
                response_content = result["result"]

                # Reference information
                top_source = result["source_documents"][0]
                page = top_source.metadata.get("page") + 1  # Correct for 0-index
                pages = top_source.metadata.get("total_pages")
                snippet = top_source.page_content

            except ConnectionError as e:
                st.exception(
                    f"Make sure Ollama container is running and `{OLLAMA_SERVER_URL=}` is correctly set."
                )
                st.stop()
            except OllamaEndpointNotFoundError as e:
                response_content = "**ERROR!** Ollama endpoint is not found. Make sure to pull the model first!"
                st.exception(response_content)
                st.stop()
            # except Exception as e:
            #     st.exception(f"Some other error. Please see exception message: {e}")
            #     st.stop()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_content
            + (
                ""
                if not page
                else f"""

        See page {page} of {pages}
        """
            ),
        }
    )
    st.rerun()
