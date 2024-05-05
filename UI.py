# Import necessary libraries
import databutton as db
import streamlit as st
import openai
import time 
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from brain import custom_search
import os
import re 

st.set_page_config(layout="wide")

st.title("Clinical Trials RAG Chatbot")

with open(r"openai_key.txt", 'r') as file:
    api_key = file.read().strip()
    
os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = api_key

# st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

prompt_template = """

        You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

        Keep your answer creative.
        
        Please take into account the previous messages as well.
        
        Make sure to citation for the answer from metadata.
            
        Reply to greetings messages.
    """

if "prompt" not in st.session_state:
    st.session_state["prompt"] = [{"role": "system", "content": prompt_template}, {"role": "user", "content": "Hi! How are you?"}, {"role": "assistant", "content": "Hello, How can I help you?"}]

prompt = st.session_state.get("prompt")

question = st.chat_input("Ask anything")

for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])
    

if question:
    

    faiss_path = r"new_db"
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.load_local(faiss_path, embeddings)
    pattern = r'\bNCT\d{8}\b'
    matches = re.findall(pattern, question)
    
    if len(matches)>0:
    
        matches_text = ", ".join(matches)
        results = custom_search(matches)
        source_knowledge = "\n".join([x.page_content for x in results])

    else:
        results = db.similarity_search(question, k=3)
        source_knowledge = "\n".join([x.page_content for x in results])

    results    
    augmented_prompt = f"""Using the contexts below, answer the query. Contexts: {source_knowledge} Query: {question}"""

    prompt.append({"role": "user", "content": augmented_prompt})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        botmsg = st.empty()

    response = []
    result = ""
    for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=prompt, stream=True):
        text = chunk.choices[0].get("delta", {}).get("content")
        if text is not None:
            response.append(text)
            result = "".join(response).strip()
            botmsg.write(result)

    prompt.pop()
    prompt.append({"role": "user", "content": question})
    prompt.append({"role": "assistant", "content": result})

    st.session_state["prompt"] = prompt
