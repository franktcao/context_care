import requests
import json
from time import sleep

import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.llms.ollama import OllamaEndpointNotFoundError

OLLAMA_SERVER_URL = "http://localhost:11434"
# OLLAMA_SERVER_URL = "http://ollama-container:11434"


# === Format page
# Fill up space
st.set_page_config(layout="wide")
# Remove whitespace from the top of the page and sidebar
st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


st.title("ContextCare")
st.write("*`Care` with the right **`Context`***")
with st.sidebar:
    uploaded = st.file_uploader("Upload document here", type=[".pdf"])

if "messages" not in st.session_state:
    st.session_state.messages = []
name = st.text_input("Name", placeholder="Enter your name here!")
if name == "":
    name = "User"

container = st.container(height=530)
for message in st.session_state.messages:
    with container:
        role = message["role"]
        display_name = "ContextCare" if role == "assistant" else name
        with st.chat_message(role):
            content = message["content"]
            display_message = f"**{display_name}:** {content}"
            st.markdown(display_message)

user_message = st.chat_input("Write your message here!")
if user_message:
    role = "user"
    st.session_state.messages.append({"role": role, "content": user_message})
    with container:
        with st.chat_message(role):
            display_name = "ContextCare" if role == "assistant" else name
            display_message = f"**{display_name}:** {user_message}"
            st.markdown(display_message)
        with st.spinner("Thinking..."):
            try:
                model_name = "tinyllama"
                llm = Ollama(model=model_name, base_url=OLLAMA_SERVER_URL)
                response_content = llm.invoke(user_message)
            except OllamaEndpointNotFoundError as e:
                response_content = "**ERROR!** Ollama endpoint is not found. Make sure to pull the model first!"
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    st.rerun()

    # st.error(response_content)
    # st.session_state.messages.append(
    #     {"role": "assistant", "content": "hi"}
    # )
    # st.rerun()
    # url = f"{OLLAMA_SERVER_URL}/api/generate"
    # headers = {
    #     "Content-Type": "application/json"
    # }

    # data = {
    #     "model": "tinyllama",
    #     "prompt": user_message,
    #     "stream": False
    # }
    # llm = Ollama(model="tinyllama", base_url=OLLAMA_SERVER_URL)
    # response_content = llm.invoke(user_message)
    # with st.spinner("Thinking..."):
