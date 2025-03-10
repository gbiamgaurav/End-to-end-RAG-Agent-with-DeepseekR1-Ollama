import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query.
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query}
Context: {document_context}
Answer:
"""

PDF_STORAGE_PATH = "document_store/pdfs/"
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")


# Create the directory if it doesn't exist
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path


def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()


def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)


def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)


def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)


def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})


# UI Configuration
st.title("AI using Deepseek R1")
st.markdown("Your AI friend")
st.markdown("-----")


# File upload section
uploaded_pdfs = st.file_uploader(
    "Upload PDF Documents here",
    type="pdf",
    help="Select one or more PDF documents for Analysis",
    accept_multiple_files=True
)

if uploaded_pdfs:
    # Loop through each uploaded PDF
    for uploaded_pdf in uploaded_pdfs:
        saved_path = save_uploaded_file(uploaded_pdf)
        raw_docs = load_pdf_documents(saved_path)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)

    st.success("Documents processed Successfully! Ask your questions below")

    user_input = st.chat_input("Enter your question")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Analyzing Document"):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)

        with st.chat_message("assistant"):
            st.write(ai_response)
