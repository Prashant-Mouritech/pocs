# Import necessary libraries
import os
import streamlit as st
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchableField, SearchField, VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration
from azure.core.credentials import AzureKeyCredential
from azure.ai.openai import OpenAIClient
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Azure Credentials
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_version = os.getenv('API_VERSION')
azure_openai_embedding_deployment = "text-embedding-ada-002"
azure_openai_chat_model = "gpt-35-turbo"
SEARCH_INDEX_NAME = "clinical_trial"
PDF_PATH = r"your_path_to_pdf.pdf"

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Step 2: Chunk the extracted text
def get_text_chunks(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(pdf_text)

# Step 3: Create Azure Search Index
def create_index():
    index_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_ADMIN_KEY))
    try:
        existing_index = index_client.get_index(SEARCH_INDEX_NAME)
        print(f"Index '{SEARCH_INDEX_NAME}' already exists.")
        return False  # Index already exists
    except Exception:
        print(f"Index '{SEARCH_INDEX_NAME}' not found. Creating a new one.")
    
    index_fields = [
        SimpleField(name="id", type="Edm.String", key=True),
        SearchableField(name="fileName", type="Edm.String"),
        SearchableField(name="content", type="Edm.String"),
        SearchField(name="contentEmbeddings", type="Collection(Edm.Single)", vector_search_dimensions=1536)
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")]
    )

    index = SearchIndex(name=SEARCH_INDEX_NAME, fields=index_fields, vector_search=vector_search)
    index_client.create_index(index)
    print(f"Index '{SEARCH_INDEX_NAME}' created successfully.")
    return True

# Step 4: Index document chunks and generate embeddings
def index_documents_to_azure_search(text_chunks):
    openai_client = OpenAIClient(azure_openai_endpoint, DefaultAzureCredential())
    search_client = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX_NAME, AzureKeyCredential(SEARCH_ADMIN_KEY))

    for i, chunk in enumerate(text_chunks):
        # Create embeddings
        response = openai_client.embeddings.create(
            input=chunk, model=azure_openai_embedding_deployment
        )
        embeddings = response.data[0].embedding

        document = {
            "id": str(i),
            "fileName": "your_pdf_file.pdf",
            "content": chunk,
            "contentEmbeddings": embeddings
        }

        # Upload the document to Azure Search
        search_client.upload_documents(documents=[document])
        print(f"Document chunk '{i}' indexed successfully.")

# Step 5: Search using embeddings
def search_embeddings(user_question):
    search_client = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX_NAME, AzureKeyCredential(SEARCH_ADMIN_KEY))
    openai_client = OpenAIClient(azure_openai_endpoint, DefaultAzureCredential())
    
    # Generate embeddings for the user question
    response = openai_client.embeddings.create(
        input=user_question, model=azure_openai_embedding_deployment
    )
    question_embedding = response.data[0].embedding

    # Search the index with the generated embedding
    results = search_client.search(search_text="", vector=question_embedding, top=5)
    return results

# Step 6: Get response
def get_chat_response(user_question):
    try:
        results = search_embeddings(user_question)
        return results
    except Exception as e:
        return f"Error generating response: {e}"

# Main application
def main():
    st.set_page_config(page_title="Chat with PDF via Azure AI", layout="wide")
    st.header("Chat with PDF using Azure OpenAI GPT-35-turbo")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question based on the PDF content")

    if user_question:
        with st.spinner("Generating a response..."):
            results = get_chat_response(user_question)
            st.session_state.chat_history.append((user_question, results))

    if st.session_state.chat_history:
        for i, (question, response) in enumerate(reversed(st.session_state.chat_history)):
            st.write(f"**Q{i+1}:** {question}")
            st.write(f"**A{i+1}:** {response}")

    if not os.path.exists("search_index"):
        with st.spinner("Processing PDF..."):
            raw_text = extract_text_from_pdf(PDF_PATH)
            text_chunks = get_text_chunks(raw_text)
            if create_index():
                index_documents_to_azure_search(text_chunks)
            st.success("PDF processed and indexed successfully.")

if __name__ == "__main__":
    main()
