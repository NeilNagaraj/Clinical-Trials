import databutton as db
import re
from io import BytesIO
from typing import Tuple, List
import pickle
import openai
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from PyPDF2 import PdfReader
import faiss
import json
from pathlib import Path
from openai.embeddings_utils import cosine_similarity


def parse_pdf(file, filename):
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output, filename


def text_to_docs(text, filename):
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc.metadata["filename"] = filename  # Add filename to metadata
            doc_chunks.append(doc)
    return doc_chunks


def docs_to_index(docs, openai_api_key):
    index = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))
    return index


def get_index_for_pdf(pdf_files, pdf_names, openai_api_key):

    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents = documents + text_to_docs(text, filename)
    index = docs_to_index(documents, openai_api_key)
    return index


def generate_responses( chat, messages, faiss_path, query):
    
    with open(r"openai_key.txt", 'r') as file:
        api_key = file.read().strip()
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.load_local(faiss_path, embeddings)

    results = db.similarity_search(query, k=3)
    source_knowledge = "\n".join([x.page_content for x in results])
    augmented_prompt = f"""Using the contexts below, answer the query. Contexts: {source_knowledge} Query: {query}"""

    prompt = {"role": "user","content":augmented_prompt}
    messages.append(prompt)
    completion = openai.ChatCompletion.create( model="gpt-3.5-turbo",messages=messages)
    result = completion.choices[0].message.content
    messages.append({"role": "assistant","content":result})

    return messages, result

def queries_similar( messages, second_query):

    first_query = messages[-2]
    prompt = f"""
    Query- I: {first_query}
    Query-II: {second_query}

    Are these queries referring to the same information? Please answer in Yes or No.
    """
    prompt_list = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt}]
    completion = openai.ChatCompletion.create( model="gpt-3.5-turbo",messages=prompt_list)
    result = completion.choices[0].message.content
    vectors = openai.Embedding.create(input = [ "Yes", "No", result], engine="text-embedding-ada-002")
    vector1 = vectors["data"][0]["embedding"]
    vector2 = vectors["data"][1]["embedding"]
    vector3 = vectors["data"][2]["embedding"]
    score1 = cosine_similarity( vector1, vector3)
    score2 = cosine_similarity( vector2, vector3)
    print("second_query ")
    print("Yes:", score1)
    print("No:", score2)
    print()
    if score1>score2:
        return "Yes"
    else:
        return "No"

def form_query( first_query, second_query):
    
    prompt = f"""
    Query- I: {first_query}
    Query-II: {second_query}

    Can you form a third query that combines the context of the first query with the second query? Just give the query.
    """
    prompt_list = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt}]
    completion = openai.ChatCompletion.create( model="gpt-3.5-turbo",messages=prompt_list)
    result = completion.choices[0].message.content
    return result

def Average(lst): 
    return sum(lst) / len(lst) 


def custom_search(matches):
    
    data = json.loads(Path("study_data.json").read_text())
    docs = []
    for protocol in data:
            if protocol["NCT_ID"] in matches:
                text = "NCT ID: " + protocol["NCT_ID"] + " "
                text += "Title: " + protocol["TITLE"] + " "
                text += "Short Title: " + protocol["SHORT_TITLE"] + " "
                text += "Sponsor: " + protocol["SPONSOR"] + " "
                text += "Detailed Eligibility: " + protocol["DETAILED_ELIGIBILITY"] + " "
                if "DESCRIPTION" in protocol:
                    text += "Description: " + protocol["DESCRIPTION"] + " "
                text += "Summary: " + protocol["SUMMARY"] + " "
                text += "Status: " + protocol["STATUS"] + " "
                if  "OUTCOME_DESCRIPTION" in protocol:
                    text += "Outcome Description: " + protocol["OUTCOME_DESCRIPTION"] + " "
                if "OUTCOME_MEASURE" in protocol:
                    text += "Outcome Measure: " + protocol["OUTCOME_MEASURE"] + " "
                if "OUTCOME_TIMEFRAME" in protocol:
                    text += "Outcome Timeframe: " + protocol["OUTCOME_TIMEFRAME"] + " "
                text += "Age Description: " + protocol["AGE_DESCRIPTION"] + " "
                if "INVESTIGATOR_NAME" in protocol:
                    text += "Investigator Name: " + protocol["INVESTIGATOR_NAME"]+ " "
                docs.append(Document(page_content=text, metadata={"source": protocol["NCT_ID"]}))
    
    return docs