
import os
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
import tempfile

#----------------------------------------------------------------------------------------------#
#--------------------------------------------LANGCHAIN-MODEL-----------------------------------#
#----------------------------------------------------------------------------------------------#

# Embedding
embedding_function = CohereEmbeddings(user_agent="PAST_YOUR_COHERE_API_KEY_HERE")

# Temporary Database
vector_store = Chroma(embedding_function=embedding_function)
document_store = InMemoryStore()

# Text Splitters
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
prompt_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)

# Relevant Content Retriever
retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=document_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# Reranking
compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# LLM
llm = Cohere(temperature=0)

# LangChain
chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=compression_retriever
)

#----------------------------------------------------------------------------------------------#
#--------------------------------------------STREAMLIT-APP-------------------------------------#
#----------------------------------------------------------------------------------------------#

# Initialize session state variables
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

# Function to handle file uploads
def upload_pdfs():
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if uploaded_files:
        temp_dir = tempfile.TemporaryDirectory()
        temp_files = []

        for file in uploaded_files:
            file_path = os.path.join(temp_dir.name, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            temp_files.append(file_path)

        documents = []
        for path in temp_files:
            loader = UnstructuredPDFLoader(path)
            documents.extend(loader.load())

        temp_dir.cleanup()  # Clean up temporary directory

        st.session_state.documents = documents

# Function to handle URL input
def input_url():
    st.session_state.url = st.sidebar.text_input("Enter URL", key="url_input")

# Function to process the input
def process():
    if st.session_state.Mode == 'pdfs':
        documents = st.session_state.documents

    elif st.session_state.Mode == 'url':
        loader = WebBaseLoader(st.session_state.url)
        documents = loader.load()
        st.session_state.documents = documents

    retriever.add_documents(documents)

# Function to handle user queries
def reply(query):
    response = chain({"query": query})
    return response["result"]

# Function to display chat history
def display_chat():
    for chat in st.session_state.chat_history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

# Function to summarize the documents
def summarize_documents():
    if not st.session_state.documents:
        return "No documents to summarize."

    full_text = " ".join([doc.page_content for doc in st.session_state.documents])
    chunks = prompt_splitter.split_text(full_text)
    summaries = []

    for chunk_text in chunks:
        summary_query = "Summarize the following text : " + chunk_text

        try:
            summary_result = llm.generate(prompts=[summary_query], max_tokens=1000)  # Adjust max_tokens as needed
            summaries.append(summary_result.generations[0][0].text)
        except Exception as e:
            summaries.append("An error occurred while generating the summary.")

    # Combine all chunk summaries
    detailed_summary = " ".join(summaries)
    return detailed_summary

# UI layout
st.set_page_config(layout="wide")  # Use wide layout

st.title("My ChatBot")

st.sidebar.header("Configuration")

option = st.sidebar.selectbox("Choose input type", ["PDFs", "URL"])

if option == "PDFs":
    st.session_state.Mode = 'pdfs'
    upload_pdfs()
elif option == "URL":
    st.session_state.Mode = 'url'
    input_url()

if st.sidebar.button("Process"):
    process()

if st.sidebar.button("Summarize"):
    summary = summarize_documents()
    st.session_state.chat_history.append({"user": "Document Summary", "bot": summary})

st.write("")  # Add some space for better separation

st.sidebar.write("")  # Add space in the sidebar

user_input = st.chat_input("Say something")

if user_input:
    response = reply(user_input)
    st.session_state.chat_history.append({"user": user_input, "bot": response})

display_chat()
