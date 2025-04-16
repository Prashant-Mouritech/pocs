import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchableField, SearchField, VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv

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
PDF_PATH = r"C:\Users\prashantvi\Desktop\POC_AZURE\azure_agent\data\DNDi-Clinical-Trial-Protocol-BENDITA-V5.pdf"

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(pdf_text)
    return chunks

def create_index():
    index_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_ADMIN_KEY))
    try:
        existing_index = index_client.get_index(SEARCH_INDEX_NAME)
        print(f"Index '{SEARCH_INDEX_NAME}' already exists.")
        return False  # Index already exists
    except Exception as e:
        print(f"Index '{SEARCH_INDEX_NAME}' not found. Creating a new one.")
    
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
    
    try:
        index_client.create_index(index)
        print(f"Index '{SEARCH_INDEX_NAME}' has been created successfully.")
        return True  # Index created successfully
    except Exception as e:
        print(f"Error creating index '{SEARCH_INDEX_NAME}': {e}")
        return False

def index_documents_to_azure_search(text_chunks):
    """Generates embeddings and indexes chunks of text into Azure Cognitive Search."""
    openai_credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(openai_credential, "https://cognitiveservices.azure.com/.default")
    client = AzureOpenAI(
        azure_deployment=azure_openai_embedding_deployment,
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_key,
        azure_ad_token_provider=token_provider if not azure_openai_key else None,
        api_version=azure_openai_version
    )

    client_search = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX_NAME, credential=AzureKeyCredential(SEARCH_ADMIN_KEY))

    for i, chunk in enumerate(text_chunks):
        document = {
            "id": str(i),
            "fileName": "sample_pdf_file.pdf",  # Replace this with the actual file name
            "content": chunk,
            "contentEmbeddings": []  # Placeholder for embeddings
        }

        # Create embeddings for the content chunk
        try:
            response = client.embeddings.create(input=chunk, model="text-embedding-ada-002")
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


def search_embeddings(user_question):
    client_search = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX_NAME, credential=AzureKeyCredential(SEARCH_ADMIN_KEY))

    # Generate embeddings for the user's question using Azure OpenAI
    try:
        openai_credential = AzureKeyCredential(azure_openai_key)
        client = AzureOpenAI(
            azure_deployment=azure_openai_embedding_deployment,
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            api_version=azure_openai_version,
        )
        
        response = client.embeddings.create(input=user_question, model=azure_openai_embedding_deployment) # GPT model
        question_embedding = response.data[0].embedding
        
        # Perform the search using the generated embeddings
        results = client_search.search(search_text="", vector=question_embedding, top=5)
        return results
    except Exception as e:
        print(f"Error generating embeddings or searching: {e}")
        return None


# def get_chat_response(user_question):
#     try:
#         response = search_embeddings(user_question)
#         return response
#     except Exception as e:
#         print(f"Error generating chat response: {e}")
#         return f"Error generating response: {e}"

def get_chat_response(user_question):
    try:
        # First, search the embeddings using the user's question
        search_results = search_embeddings(user_question)
        
        # Prepare a system prompt based on search results (can be customized)
        prompt = "The following is the context based on the search results: \n\n"
        for result in search_results:
            prompt += f"{result['content']}\n\n"
        
        # Add the user's question to the prompt
        prompt += f"User question: {user_question}\n\n"
        
        # Use GPT-35-turbo for chat completion
        try:
            client = AzureOpenAI(
                azure_deployment=azure_openai_chat_model,
                azure_endpoint=azure_openai_endpoint,
                api_key=azure_openai_key,
                api_version=azure_openai_version,
            )
            response = client.chat.completions.create(
                model="gpt-35-turbo",
                messages=[{"role": "system", "content": prompt}]
            )
            output = response.choices[0].message.content

            # Return the response generated by GPT-35-turbo
            return {
                "response": output,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating response: {e}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    st.set_page_config(page_title="Chat with PDF via Azure AI", layout="wide")
    st.header("Chat with PDF using Azure OpenAI GPT-35-turbo")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

    user_question = st.text_input("Ask a question based on the PDF content")

    if user_question:
        with st.spinner("Generating a response..."):
            results = get_chat_response(user_question)
            st.session_state.chat_history.append((user_question, results))

    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, (question, response) in enumerate(reversed(st.session_state.chat_history)):
            st.write(f"**Q{i+1}:** {question}")
            st.write(f"**A{i+1}:** {response}")
            st.write("---")

    if not os.path.exists("search_index"):
        with st.spinner("Processing PDF..."):
            raw_text = extract_text_from_pdf(PDF_PATH)
            text_chunks = get_text_chunks(raw_text)
            if create_index():
                index_documents_to_azure_search(text_chunks)
            st.success("PDF processed and indexed successfully.")

if __name__ == "__main__":
    main()


