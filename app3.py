from typing import Set

from main_final3 import run_llm
import streamlit as st
from streamlit_chat import message

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = ''


if 'llm' not in st.session_state:
    st.session_state['llm'] = ''

if 'db' not in st.session_state:
    st.session_state['db'] = ''

if 'loaded_db' not in st.session_state:
    st.session_state['loaded_db'] = '' 

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("gita4.jpg")
    
st.header("Gita GPT - Ask Gita related questions")
st.write("By Ravi Shankar Prasad")
st.write("https://www.linkedin.com/in/ravi-shankar-prasad-371825101/")


prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")
#prompt = st.chat_input("Prompt")
if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response.."):
        # generated_response = run_llm(
        #     query=prompt, chat_history=st.session_state["chat_history"]
        # )
        generated_response = run_llm(
        query=prompt#, chat_history=st.session_state["chat_history"]
        )
        #st.write(generated_response)
        # sources = (
        #     [doc for doc in generated_response["context"]]
        # )

        # formatted_response = (
        #     f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        # )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(generated_response["answer"])
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["answer"]))


if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)
        message(generated_response)
