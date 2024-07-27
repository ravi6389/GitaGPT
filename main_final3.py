import os

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from pypdf import PdfReader
from langchain import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
#from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from typing import Any, Dict, List

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

from langchain.chains.retrieval import create_retrieval_chain

from langchain.document_loaders import CSVLoader

from langchain.indexes import VectorstoreIndexCreator

from sentence_transformers import SentenceTransformer, util
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.llms import Ollama

import transformers
import torch
from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForCausalLM


#from hugchat import hugchat
from typing import Any, Dict, List
import streamlit as st

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = ''

if 'llm' not in st.session_state:
    st.session_state['llm'] = ''

if 'db' not in st.session_state:
    st.session_state['db'] = ''

if 'loaded_db' not in st.session_state:
    st.session_state['loaded_db'] =''

if 'run_once' not in st.session_state:
    st.session_state['run_once'] = 0




load_dotenv()

if (st.session_state['run_once'] == 0):
   
    st.session_state['run_once'] = 1

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
#def run_llm(query: str):
    
    if (st.session_state['embeddings'] ==''):
   
        
        embeddings = HuggingFaceEmbeddings()
       
        
        st.session_state['embeddings'] = embeddings
    else:
        embeddings = st.session_state['embeddings']
   


   
    if(st.session_state['loaded_db'] == ''):
    
        loaded_db = FAISS.load_local('becki_fast',\
    embeddings, allow_dangerous_deserialization=True)
        st.session_state['loaded_db'] = loaded_db
    
    else:
        loaded_db = st.session_state['loaded_db']
    
    
    if(st.session_state['llm'] ==''):
       
        llm = Ollama(model= 'phi', temperature = 0.8)

        
        st.session_state['llm'] = llm
        st.write(st.session_state['llm'])
    else:
        llm =  st.session_state['llm']

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=loaded_db.as_retriever(), prompt=rephrase_prompt
    )
   
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query})#, "chat_history": chat_history})
 
    return result



   
