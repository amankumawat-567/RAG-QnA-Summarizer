
import os
import tempfile
from langchain.document_loaders import UnstructuredPDFLoader, WebBaseLoader
from langchain_community.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
from langchain.chains import RetrievalQA
import streamlit as st

def Create_chain(COHERE_API_KEY):
    os.environ["COHERE_API_KEY"] = COHERE_API_KEY
    embedding_function = CohereEmbeddings(user_agent=COHERE_API_KEY)
    vector_store = Chroma(embedding_function=embedding_function)
    document_store = InMemoryStore()
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    prompt_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=document_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    compressor = CohereRerank(model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    llm = Cohere(temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=compression_retriever
    )
    return chain, retriever, prompt_splitter, llm, document_store, vector_store

st.set_page_config(layout="wide")
st.title("My ChatBot")
st.sidebar.header("Configuration")

COHERE_API_KEY = st.sidebar.text_input("Enter Cohere API Key", type="password") #Q11gNLtRpb3rmCMKzmsCasVcEMJoz1ZEKizgN7jC
if COHERE_API_KEY and 'chain' not in st.session_state:
    st.session_state.chain, st.session_state.retriever, st.session_state.prompt_splitter, \
    st.session_state.llm, st.session_state.document_store, st.session_state.vector_store = Create_chain(COHERE_API_KEY)

if 'pdfs' not in st.session_state:
    st.session_state.pdfs = []
if 'Mode' not in st.session_state:
    st.session_state.Mode = None
if 'url' not in st.session_state:
    st.session_state.url = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'status' not in st.session_state:
    st.session_state.status = ""

def upload_pdfs():
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    if uploaded_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_files = []
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                temp_files.append(file_path)
            documents = []
            for path in temp_files:
                try:
                    loader = UnstructuredPDFLoader(path)
                    documents.extend(loader.load())
                except Exception as e:
                    st.error(f"Error loading document {path}: {e}")
            st.session_state.documents = documents

def input_url():
    st.session_state.url = st.sidebar.text_input("Enter URL", key="url_input")

def process():
    if st.session_state.Mode == 'pdfs':
        documents = st.session_state.documents
    elif st.session_state.Mode == 'url':
        try:
            loader = WebBaseLoader(st.session_state.url)
            documents = loader.load()
            st.session_state.documents = documents
        except Exception as e:
            st.error(f"Error loading URL: {e}")
            return
    st.session_state.retriever.add_documents(documents)

def reply(query):
    response = st.session_state.chain({"query": query})
    return response["result"]

def display_chat():
    for chat in st.session_state.chat_history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

def summarize_documents():
    if not st.session_state.documents:
        return "No documents to summarize."
    full_text = " ".join([doc.page_content for doc in st.session_state.documents])
    chunks = st.session_state.prompt_splitter.split_text(full_text)
    summaries = []
    for chunk_text in chunks:
        summary_query = "Summarize the following text: " + chunk_text
        try:
            summary_result = st.session_state.llm.generate(prompts=[summary_query], max_tokens=1000)
            summaries.append(summary_result.generations[0][0].text)
        except Exception as e:
            summaries.append("An error occurred while generating the summary.")
    detailed_summary = " ".join(summaries)
    return detailed_summary

option = st.sidebar.selectbox("Choose input type", ["PDFs", "URL"])

if option == "PDFs":
    st.session_state.Mode = 'pdfs'
    upload_pdfs()
elif option == "URL":
    st.session_state.Mode = 'url'
    input_url()

if st.sidebar.button("Process"):
    with st.spinner("Processing documents..."):
        process()

if st.sidebar.button("Summarize"):
    summary = summarize_documents()
    st.session_state.chat_history.append({"user": "Document Summary", "bot": summary})

user_input = st.chat_input("Say something")

if user_input:
    response = reply(user_input)
    st.session_state.chat_history.append({"user": user_input, "bot": response})

display_chat()
