# Import necessary libraries
import databutton as db
import streamlit as st
import openai
import time 
from brain import get_index_for_pdf
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import os
from brain import queries_similar,Average, form_query

st.set_page_config(layout="wide")

st.title("Clinical Trials RAG Chatbot")

with open(r"openai_key.txt", 'r') as file:
    api_key = file.read().strip()
    
os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = api_key

# st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

with st.sidebar:
    st.markdown('<p style="font-size: 20px;"><strong>Please Select The Vector Database:</strong></p>', unsafe_allow_html=True)
    choice = st.radio("",["Central Database", "Trial Database", "Policy Database", "Upload Your Own Documents"], captions = [ "This database is a collection of both policy & trials information.", "This database comprises of trials information.", "This database comprises policy documents.", "This will create a database of provided documents."])

if choice == "Policy Database":

    st.session_state["xml_loaded"]=False
    st.session_state["main_db_loaded"]=False
    st.session_state["any_loaded"]=False

    prompt_template = """

        You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

        Keep your answer creative.
        
        Please take into account the previous messages as well.
        
        Make sure to citation for the answer from metadata.
            
        Reply to greetings messages & say "Apologies, I'm unable to provide a response to that inquiry at this moment. For further assistance, please feel free to reach out to us via phone at 714-456-7890 or visit our website at ucihealth.org. We'll be happy to help you there." if the text is irrelevant.

    """

    if "policy_loaded" not in st.session_state or st.session_state["policy_loaded"]==False:
        with st.spinner("Loading Database"):
            st.session_state["policy_loaded"]=True
            st.session_state["prompt"] = [{"role": "system", "content": prompt_template}, {"role": "user", "content": "Hi! How are you?"}, {"role": "assistant", "content": "Hello, How can I help you?"}]
            time.sleep(2)

    prompt = st.session_state.get("prompt")
    
    question = st.chat_input("Ask anything")

    for message in prompt:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])
        

    if question:
        
        faiss_path = r"C:\Users\gaura\OneDrive\Documents\DTF\clinical-trial-matching-master\Clinical-Trails\Vector_DB\policies"
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        db = FAISS.load_local(faiss_path, embeddings)
        answer = "No" if len(prompt)<=2 else queries_similar(prompt, question)
       
        if not "results" in st.session_state or answer == "No":
            st.session_state["results"] = db.similarity_search_with_score(question, k=5)
        
        results,scores = [],[]
        for result in st.session_state["results"]:
            results.append(result[0])
            scores.append(result[1])

        score = Average(scores)
        source_knowledge = "\n".join([x.page_content for x in results])
        augmented_prompt = f"""Using the contexts below, answer the query. Contexts: {source_knowledge} Query: {question}"""

        prompt.append({"role": "user", "content": augmented_prompt})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            botmsg = st.empty()

        response = []
        result = ""

        if score<0.40:
            for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=prompt, stream=True):
                text = chunk.choices[0].get("delta", {}).get("content")
                if text is not None:
                    response.append(text)
                    result = "".join(response).strip()
                    botmsg.write(result)
        else:
            text = "Apologies, I'm unable to provide a response to that inquiry at this moment. For further assistance, please feel free to reach out to us via phone at 714-456-7890 or visit our website at ucihealth.org. We'll be happy to help you there."
            response.append(text)
            result = "".join(response).strip()
            botmsg.write(result)


        prompt.pop()
        prompt.append({"role": "user", "content": question})
        prompt.append({"role": "assistant", "content": result})

        st.session_state["prompt"] = prompt

elif choice =="Trial Database":

    st.session_state["policy_loaded"]=False
    st.session_state["main_db_loaded"]=False
    st.session_state["any_loaded"]=False


    prompt_template = """

        You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

        Keep your answer creative.
        
        Please take into account the previous messages as well.
        
        Make sure to citation for the answer from metadata.
            
        Reply to greetings messages & say "Apologies, I'm unable to provide a response to that inquiry at this moment. For further assistance, please feel free to reach out to us via phone at 714-456-7890 or visit our website at ucihealth.org. We'll be happy to help you there." if the text is irrelevant.

    """

    if "xml_loaded" not in st.session_state or st.session_state["xml_loaded"]==False:
        with st.spinner("Loading Database"):
            st.session_state["xml_loaded"]=True
            st.session_state["prompt"] = [{"role": "system", "content": prompt_template}, {"role": "user", "content": "Hi! How are you?"}, {"role": "assistant", "content": "Hello, How can I help you?"}]
            time.sleep(2)

    prompt = st.session_state.get("prompt")
    
    question = st.chat_input("Ask anything")

    for message in prompt:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])
        

    if question:

        faiss_path = r"C:\Users\gaura\OneDrive\Documents\DTF\clinical-trial-matching-master\Clinical-Trails\Vector_DB\xml_db"
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        db = FAISS.load_local(faiss_path, embeddings)
        answer = "No" if len(prompt)<=2 else queries_similar(prompt, question)

        if not "results" in st.session_state or answer == "No":
            st.session_state["results"] = db.similarity_search_with_score(question, k=5)
        
        results,scores = [],[]
        for result in st.session_state["results"]:
            results.append(result[0])
            scores.append(result[1])

        score = Average(scores)
        source_knowledge = "\n".join([x.page_content for x in results])
        augmented_prompt = f"""Using the contexts below, answer the query. Contexts: {source_knowledge} Query: {question}"""

        prompt.append({"role": "user", "content": augmented_prompt})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            botmsg = st.empty()

        response = []
        result = ""

        if score<0.40:
            for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=prompt, stream=True):
                text = chunk.choices[0].get("delta", {}).get("content")
                if text is not None:
                    response.append(text)
                    result = "".join(response).strip()
                    botmsg.write(result)
        else:
            text = "Apologies, I'm unable to provide a response to that inquiry at this moment. For further assistance, please feel free to reach out to us via phone at 714-456-7890 or visit our website at ucihealth.org. We'll be happy to help you there."
            response.append(text)
            result = "".join(response).strip()
            botmsg.write(result)

        prompt.pop()
        prompt.append({"role": "user", "content": question})
        prompt.append({"role": "assistant", "content": result})

        st.session_state["prompt"] = prompt

elif choice == "Central Database":
    
    st.session_state["policy_loaded"]=False
    st.session_state["xml_loaded"]=False
    st.session_state["any_loaded"]=False

    prompt_template = """

        You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

        Keep your answer creative and elaborate.
        
        Please take into account the previous messages as well.
        
        Don't provide the citations
            
        Reply to greetings messages.

    """

    if "main_db_loaded" not in st.session_state or st.session_state["main_db_loaded"]==False:
        with st.spinner("Loading Database"):
            st.session_state["main_db_loaded"]=True
            st.session_state["prompt"] = [{"role": "system", "content": prompt_template}, {"role": "user", "content": "Hi! How are you?"}, {"role": "assistant", "content": "Hello, How can I help you?"}]
            time.sleep(2)

    prompt = st.session_state.get("prompt")
    
    question = st.chat_input("Ask anything")

    for message in prompt:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
    if question:

        faiss_path = r"C:\Users\gaura\OneDrive\Documents\DTF\clinical-trial-matching-master\Clinical-Trails\Vector_DB\main_db"
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        db = FAISS.load_local(faiss_path, embeddings)
        
        answer = "No" if len(prompt)<=2 else queries_similar(prompt, question)

        if answer == "Yes":
            st.session_state['question'] = form_query(st.session_state['question'], question)
            print(st.session_state['question'])
            st.session_state["results"] = db.similarity_search_with_score(st.session_state["question"], k=5)

        elif not "results" in st.session_state or answer == "No":
            st.session_state["question"] = question
            st.session_state["results"] = db.similarity_search_with_score(question, k=5)
        
        results,scores = [],[]
        for result in st.session_state["results"]:
            results.append(result[0])
            scores.append(result[1])

        score = Average(scores)
        print(score)
        print()
        source_knowledge = "\n".join([x.page_content for x in results])
        augmented_prompt = f"""Using the contexts below, answer the query. Contexts: {source_knowledge} Query: {st.session_state["question"]}"""
        prompt.append({"role": "user", "content": augmented_prompt})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            botmsg = st.empty()

        response = []
        result = ""

        if score<0.45:
            for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=prompt, stream=True):
                text = chunk.choices[0].get("delta", {}).get("content")
                if text is not None:
                    response.append(text)
                    result = "".join(response).strip()
                    botmsg.write(result)
        else:
            text = "I'm unable to assist with your inquiry. For further assistance, please reach out to UC Irvine's Center for Clinical Research (CCR) at clinicalresearch@hs.uci.edu or call 949-824-9320. "
            response.append(text)
            result = "".join(response).strip()
            botmsg.write(result)

        prompt.pop()
        prompt.append({"role": "user", "content": question})
        prompt.append({"role": "assistant", "content": result})

        st.session_state["prompt"] = prompt
        

elif choice == "Upload Your Own Documents":
    
    st.session_state["policy_loaded"]=False
    st.session_state["xml_loaded"]=False
    st.session_state["main_db_loaded"]=False

    if "any_loaded" not in st.session_state or st.session_state["any_loaded"]==False:
        st.session_state["prompt"] = [{"role": "system", "content": "none"}]
        st.session_state["any_loaded"]=True

    @st.cache_data
    def create_vectordb(files, filenames):
        # Show a spinner while creating the vectordb
        with st.spinner("Vector database"):
            vectordb = get_index_for_pdf(
                [file.getvalue() for file in files], filenames, openai.api_key
            )
        return vectordb


    # Upload PDF files using Streamlit's file uploader
    pdf_files = st.file_uploader("", type="pdf", accept_multiple_files=True)

    # If PDF files are uploaded, create the vectordb and store it in the session state
    if pdf_files:
        pdf_file_names = [file.name for file in pdf_files]
        st.session_state["vectordb"] = create_vectordb(pdf_files, pdf_file_names)

    # Define the template for the chatbot prompt
    prompt_template = """
        You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

        Keep your answer creative.
        
        The evidence are the context of the pdf extract with metadata. 
        
        Carefully focus on the metadata specially 'filename' and 'page' whenever answering.
        
        Make sure to add filename and page number at the end of sentence you are citing to.
            
        Reply "Not applicable" if text is irrelevant.
        
        The PDF content is:
        {pdf_extract}
    """

    # Get the current prompt from the session state or set a default value
    prompt = st.session_state.get("prompt")

    # Display previous chat messages

    for message in prompt:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Get the user's question using Streamlit's chat input
    question = st.chat_input("Ask anything")

    # Handle the user's question
    if question:
        vectordb = st.session_state.get("vectordb", None)
        if not vectordb:
            with st.message("assistant"):
                st.write("You need to provide a PDF")
                st.stop()

        # Search the vectordb for similar content to the user's question
        search_results = vectordb.similarity_search_with_score(question, k=5)
        # search_results
        pdf_extract = "/n ".join([result.page_content for result in search_results])
        # Update the prompt with the pdf extract
        prompt[0] = {
            "role": "system",
            "content": prompt_template.format(pdf_extract=pdf_extract),
        }
        # Add the user's question to the prompt and display it
        prompt.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Display an empty assistant message while waiting for the response
        with st.chat_message("assistant"):
            botmsg = st.empty()

        # Call ChatGPT with streaming and display the response as it comes
        response = []
        result = ""
        for chunk in openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=prompt, stream=True
        ):
            text = chunk.choices[0].get("delta", {}).get("content")
            if text is not None:
                response.append(text)
                result = "".join(response).strip()
                botmsg.write(result)

        # Add the assistant's response to the prompt
        prompt.append({"role": "assistant", "content": result})

        # Store the updated prompt in the session state
        st.session_state["prompt"] = prompt
       
