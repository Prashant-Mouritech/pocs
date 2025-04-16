import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchableField, SearchField, VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
import openai  # Updated import for OpenAI
from dotenv import load_dotenv
from openai import AzureOpenAI
import PyPDF2

# Load environment variables
load_dotenv()

# Azure OpenAI and Azure Search credentials
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_version = os.getenv('API_VERSION')
azure_openai_embedding_deployment = "text-embedding-ada-002"
azure_openai_chat_model = "gpt-35-turbo"

# Constants
SEARCH_INDEX_NAME = "clinical_trial"
PDF_PATH = r"C:\Users\prashantvi\Desktop\POC_AZURE\azure_agent\data\DNDi-Clinical-Trial-Protocol-BENDITA-V5.pdf"  # Set your PDF file path here


def extract_text_from_pdf(pdf_path): # Working
    """Extracts text from a PDF file using PyPDF2."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""  # Ensure it handles None if text extraction fails
    return text

# Step 2: Get text chunks
def get_text_chunks(pdf_text): # Working
    """Splits extracted PDF text into smaller chunks for better indexing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(pdf_text)
    return chunks

# Step 3: Create the Azure Search Index
def create_index(): # Working
    """Creates an Azure Cognitive Search index with vector search."""
    index_fields = [
        SimpleField(name="id", type="Edm.String", key=True, filterable=True, sortable=True),
        SearchableField(name="fileName", type="Edm.String", filterable=True),
        SearchableField(name="content", type="Edm.String", filterable=True),
        SearchField(name="contentEmbeddings", type="Collection(Edm.Single)", vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile")
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")]
    )

    index = SearchIndex(name=SEARCH_INDEX_NAME, fields=index_fields, vector_search=vector_search)
    index_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_ADMIN_KEY))

    try:
        existing_index = index_client.get_index(SEARCH_INDEX_NAME)
        print(f"Index '{SEARCH_INDEX_NAME}' already exists. Deleting the existing index...")
        index_client.delete_index(SEARCH_INDEX_NAME)
        print(f"Index '{SEARCH_INDEX_NAME}' has been deleted.")
    except Exception as e:
        print(f"Index '{SEARCH_INDEX_NAME}' not found.")

    try:
        print(f"Creating index '{SEARCH_INDEX_NAME}'...")
        index_client.create_index(index)
        print(f"Index '{SEARCH_INDEX_NAME}' has been created successfully.")
    except Exception as e:
        print(f"Error creating index '{SEARCH_INDEX_NAME}': {e}")



# Step 4: Index Documents with Embeddings to Azure Cognitive Search (Updated code)
def index_documents_to_azure_search(text_chunks):
    """Generates embeddings and indexes chunks of text into Azure Cognitive Search."""
    openai_credential = AzureKeyCredential(azure_openai_key)  # Change to use the key directly
    client = AzureOpenAI(
        azure_deployment=azure_openai_embedding_deployment,
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_key,
        api_version=azure_openai_version,
    )

    client_search = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX_NAME, credential=AzureKeyCredential(SEARCH_ADMIN_KEY))

    for i, chunk in enumerate(text_chunks):
        document = {
            "id": str(i),
            "fileName": "DNDi-Clinical-Trial-Protocol-BENDITA-V5.pdf",  # Replace this with the actual file name
            "content": chunk,
            "contentEmbeddings": []  # Placeholder for embeddings
        }

        # Create embeddings for the content chunk
        try:
            response = client.embeddings.create(input=chunk, model=azure_openai_embedding_deployment)
            embeddings = response.data[0].embedding
            document["contentEmbeddings"] = embeddings  # Store the embeddings
        except Exception as e:
            print(f"Error generating embeddings for chunk {i}: {e}")

        # Upload the document with embeddings to Azure Search
        try:
            client_search.upload_documents(documents=[document])
            print(f"Document chunk '{document['id']}' indexed successfully.")
        except Exception as e:
            print(f"Failed to index document chunk '{document['id']}': {e}")
# Step 5: Chatbot Functionality
def get_chat_response(user_question):
    """Generates a response from Azure OpenAI GPT-35-turbo based on the user's question."""
    try:
        openai.api_key = azure_openai_key  # Set the API key

        # Construct the prompt for the GPT-35-turbo model
        response = openai.ChatCompletion.create(
            model=azure_openai_chat_model,
            messages=[{"role": "user", "content": user_question}]
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"Error generating chat response: {e}")
        return f"Error generating response: {e}"



# Step 6: Streamlit UI
def main():
    st.set_page_config(page_title="Chat with PDF via Azure AI", layout="wide")
    st.header("Chat with PDF using Azure OpenAI GPT-35-turbo")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

    # Input box for user question
    user_question = st.text_input("Ask a question based on the PDF content")

    if user_question:
        with st.spinner("Generating a response..."):
            response = get_chat_response(user_question)
            st.session_state.chat_history.append((user_question, response))

    # Display chat history in reverse order (most recent first)
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, (question, response) in enumerate(reversed(st.session_state.chat_history)):
            st.write(f"**Q{i+1}:** {question}")
            st.write(f"**A{i+1}:** {response}")
            st.write("---")

    # Load PDF and process (indexing)
    if not os.path.exists("search_index"):
        with st.spinner("Processing PDF..."):
            raw_text = extract_text_from_pdf(PDF_PATH)  # Use Langchain to extract text from PDF
            text_chunks = get_text_chunks(raw_text)
            create_index()
            index_documents_to_azure_search(text_chunks)
            st.success("PDF processed and indexed successfully.")


if __name__ == "__main__":
    main()
