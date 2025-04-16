import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Path to your PDF file
PDF_PATH = r"C:\Users\prashantvi\Desktop\POC_GOOGLE_GEMINI_PRO\chatmultipledocuments\DNDi-Clinical-Trial-Protocol-BENDITA-V5.pdf"  # Update this path accordingly

# Function to extract text from the PDF file
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split the extracted text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to store text embeddings in a FAISS index
def get_vector_store(text_chunks):
    # Initialize the embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Example model

    # Collect embeddings and document metadata
    embeddings = []
    documents = []
    
    for i, chunk in enumerate(text_chunks):
        embedding = model.encode(chunk)
        
        # Store metadata and embedding
        document = {
            "id": str(i),
            "fileName": "Document_Name.pdf",  # Replace with the actual file name or relevant identifier
            "content": chunk,
            "contentEmbeddings": embedding.tolist()  # Store as list for serialization
        }
        embeddings.append(embedding)
        documents.append(document)
    
    # Convert embeddings to numpy array for FAISS
    embeddings = np.array(embeddings).astype("float32")
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)  # Using L2 distance
    faiss_index.add(embeddings)  # Add embeddings to FAISS index
    
    # Save FAISS index and metadata locally
    faiss.write_index(faiss_index, "faiss_index.bin")
    with open("faiss_documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    print("FAISS index and metadata stored successfully.")

# Function to load the FAISS index and perform similarity search
def load_vector_store():
    # Load FAISS index
    faiss_index = faiss.read_index("faiss_index.bin")

    # Load metadata
    with open("faiss_documents.pkl", "rb") as f:
        documents = pickle.load(f)

    return faiss_index, documents

# Function to perform similarity search
def similarity_search(query, faiss_index, documents, model):
    query_embedding = model.encode(query).astype("float32")
    _, indices = faiss_index.search(np.array([query_embedding]), k=5)  # Return top 5 results

    # Retrieve documents based on indices
    results = [documents[idx] for idx in indices[0]]
    return results

# Main function to handle user input and generate responses
def main():
    st.set_page_config(page_title="Chat With PDF", layout="wide")
    st.header("Chat with PDF using Local FAISS Index")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

    # User input section for questions
    user_question = st.text_input("Ask a Question from the PDF Files")

    # Load embeddings model and FAISS index only once
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists("faiss_index.bin"):
        with st.spinner("Processing PDF..."):
            raw_text = get_pdf_text(PDF_PATH)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("PDF processing completed.")
    else:
        faiss_index, documents = load_vector_store()

    # Generate a response if the user provides a question
    if user_question:
        with st.spinner("Generating a response..."):
            relevant_docs = similarity_search(user_question, faiss_index, documents, model)
            
            # Combine content from relevant docs as the context for the answer
            context = "\n\n".join([doc["content"] for doc in relevant_docs])

            # Display the question and answer
            st.session_state.chat_history.append((user_question, context))
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, (question, response) in enumerate(reversed(st.session_state.chat_history)):
            st.write(f"**Q{i+1}:** {question}")
            st.write(f"**A{i+1}:** {response}")
            st.write("---")  # Divider for readability

if __name__ == "__main__":
    main()
