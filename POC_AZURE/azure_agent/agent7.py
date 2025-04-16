# Import necessary libraries
import os
import streamlit as st
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField, 
    VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration
)
# from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchableField, SearchField,
#  VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration

from azure.core.credentials import AzureKeyCredential
# from azure.ai.openai import OpenAIClient
from openai import AzureOpenAI
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
azure_openai_version = os.getenv('AZURE_OPENAI_VERSION')
azure_openai_embedding_deployment = "text-embedding-ada-002"
azure_openai_chat_model = "gpt-35-turbo"
SEARCH_INDEX_NAME = "clinical_trial"  # Update index name if needed
PDF_PATH = r"path_to_your_pdf.pdf"  # Update with your PDF path

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

# Step 3: Create Azure Search Index (similar to FAISS in Google Gemini)
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

# Step 4: Index document chunks and generate embeddings using Azure OpenAI
def index_documents_to_azure_search(text_chunks):
    # openai_client = OpenAIClient(azure_openai_endpoint, DefaultAzureCredential())
    openai_client = AzureOpenAI(
        azure_deployment=azure_openai_embedding_deployment,
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_key,
        azure_ad_token_provider=token_provider if not azure_openai_key else None,
        api_version=azure_openai_version
    )
    
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

        # Upload the document to Azure Search (similar to FAISS indexing in Gemini)
        search_client.upload_documents(documents=[document])
        print(f"Document chunk '{i}' indexed successfully.")

# Step 5: Search using embeddings (similar to FAISS similarity search in Gemini)
def search_embeddings(user_question):
    search_client = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX_NAME, AzureKeyCredential(SEARCH_ADMIN_KEY))
    # openai_client = OpenAIClient(azure_openai_endpoint, DefaultAzureCredential())
    openai_client = AzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            deployment=azure_openai_chat_model,
        )
    # Generate embeddings for the user question (similar to Gemini embedding search)
    response = openai_client.embeddings.create(
        input=user_question, model=azure_openai_embedding_deployment
    )
    question_embedding = response.data[0].embedding

    # Search the index with the generated embedding
    results = search_client.search(search_text="", vector=question_embedding, top=5)
    return results

# Step 6: Chat response using the same prompt style as Google Gemini
def get_chat_response(user_question):
    try:
        results = search_embeddings(user_question)
        relevant_docs = [doc['content'] for doc in results]
        
        # Combine the retrieved documents into a single context for the GPT model
        context = "\n".join(relevant_docs)

        # Custom prompt template similar to Google Gemini
        custom_prompt = f"""
        Answer the question as detailed as possible from the provided context. Make sure to provide all relevant details.
        If the answer is not in the provided context, respond with: "The answer is not available in the context."

        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        # Set up chat prompt for GPT (same logic as your Gemini chat system)
        # openai_client = OpenAIClient(azure_openai_endpoint, DefaultAzureCredential())
        openai_client = AzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            deployment=azure_openai_chat_model,
        )
        
        chat_response = openai_client.completions.create(
            model=azure_openai_chat_model,
            prompt=custom_prompt,
            max_tokens=500
        )

        return chat_response.choices[0].text.strip()
    except Exception as e:
        return f"Error generating response: {e}"

# Main application (same structure as Google Gemini Streamlit app)
def main():
    st.set_page_config(page_title="Chat with PDF via Azure AI", layout="wide")
    st.header("Chat with PDF using Azure OpenAI GPT-35-turbo")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input section
    user_question = st.text_input("Ask a question based on the PDF content")

    if user_question:
        with st.spinner("Generating a response..."):
            results = get_chat_response(user_question)
            st.session_state.chat_history.append((user_question, results))

    # Display chat history
    if st.session_state.chat_history:
        for i, (question, response) in enumerate(reversed(st.session_state.chat_history)):
            st.write(f"**Q{i+1}:** {question}")
            st.write(f"**A{i+1}:** {response}")

    # Backend processing for PDF and Azure Cognitive Search setup (similar to Gemini and FAISS)
    if not os.path.exists("search_index"):
        with st.spinner("Processing PDF..."):
            raw_text = extract_text_from_pdf(PDF_PATH)
            text_chunks = get_text_chunks(raw_text)
            if create_index():
                index_documents_to_azure_search(text_chunks)
            st.success("PDF processed and indexed successfully.")

if __name__ == "__main__":
    main()
