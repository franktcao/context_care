import streamlit as st


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
        with st.chat_message(role):
            display_name = "ContextCare" if role == "assistant" else name
            content = message["content"]
            display_message = f"**{display_name}:** {content}"
            st.markdown(display_message)

user_message = st.chat_input("Write your message here!")
if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hi, how can I help you today?"}
    )
    st.rerun()
