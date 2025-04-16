import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import google.generativeai as genai
import logging
from time import sleep

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables for the Google API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API key is not set. Check your .env file.")
genai.configure(api_key=api_key)

# Path to your folder containing PDFs
PDF_FOLDER_PATH = r"C:\\Users\\prashantvi\\Desktop\\chatwithmultiple_docs\\Data"  # Update this path accordingly

# Function to extract text from all PDFs in a folder
def extract_text_from_folder(folder_path):
    pdf_texts = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(pdf_path)
            text = " ".join([page.extract_text() for page in pdf_reader.pages])
            pdf_texts.append((filename, text))  # Store filename with text
    return pdf_texts

# Function to split the extracted text into manageable chunks
def get_text_chunks(pdf_texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for filename, text in pdf_texts:
        for chunk in text_splitter.split_text(text):
            chunks.append((filename, chunk))  # Include filename with each chunk
    return chunks

# Retry helper function
def safe_embed_documents(embedding_model, documents, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return embedding_model.embed_documents(documents)
        except Exception as e:
            logging.error(f"Embedding attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                sleep(delay)
            else:
                raise e  # Raise the error after exhausting retries

# Function to store text embeddings in a FAISS index with document information
def get_vector_store(text_chunks):
    # Initialize the Google Generative AI embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", timeout=120)

    text_data = []
    metadata = []

    for i, (filename, chunk) in enumerate(text_chunks):
        metadata.append({
            "id": str(i),
            "fileName": filename,
            "content": chunk
        })
        text_data.append(chunk)

    try:
        # Embed documents in smaller batches
        text_embeddings = []
        batch_size = 10
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i + batch_size]
            # Get embeddings for the current batch
            text_embeddings.extend(safe_embed_documents(embeddings, batch))

        # Create the FAISS vector store using the embedding model
        vector_store = FAISS.from_texts(text_data, embedding=embeddings, metadatas=metadata)
        vector_store.save_local("faiss_index")
        logging.info("FAISS index with document metadata created and saved locally.")
    except Exception as e:
        logging.error(f"Failed to create FAISS index: {e}")


# Function to set up the conversational chain with Google Gemini
def get_conversational_chain():
    # Based on the provided context, answer the following question with as much detail as possible. 
    # Include references to the relevant sections and documents.
    prompt_template = """
    Based on the provided context, answer the following question like if eg. [context is Warranty Duration the ans will be 2 year]. 
    Include references to the relevant sections and documents.

    Context:
    {context}

    Question:
    {question}

    Answer:
    - Detailed Answer:
    - Section Reference: [Include the section reference]
    - Related Document(s): [Include the document name(s)]
    """

    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, streaming=True)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

# Function to handle user input and process responses
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)

    # Check if any documents were retrieved
    if not docs:
        st.write("No relevant documents found for the query.")
        return "No relevant context available.", []

    context = "\n\n".join([doc.page_content for doc in docs])
    chain = get_conversational_chain()

    # Log context for debugging
    # print("Retrieved Context:", context)

    # streamed_response = []
    # for chunk in chain.stream({"context": context, "question": user_question}):
    #     if isinstance(chunk, dict):
    #         content = chunk.get("content", "")
    #         streamed_response.append(content)
    #         st.write(content, end="", flush=True)
    #     else:
    #         st.write("Unexpected response format:", chunk)

    # response_text = "".join(streamed_response)

    # # Ensure response_text is not empty
    # if not response_text.strip():
    #     response_text = "The model did not provide an answer."

    # relevant_docs = set(doc.metadata["fileName"] for doc in docs)
    # relevant_document_references = [
    #     {"doc_name": doc, "reference": f"Document: {doc}, Reference Section: Some section details"}
    #     for doc in relevant_docs
    # ]

    # return response_text, relevant_document_references
    # Generate a response using the chain and relevant documents
    response = chain({"context": context, "question": user_question})

    # Verify the response format and extract the output
    if "output_text" in response:
        response_text = response["output_text"]
    elif isinstance(response, dict):
        response_text = next(iter(response.values()), "No response generated.")
    else:
        response_text = "No response generated."

    # Include document names in the response
    relevant_docs = set(doc.metadata["fileName"] for doc in docs)

    # For each document, match the information with the relevant documents (similar to your examples)
    relevant_document_references = []
    for doc in relevant_docs:
        relevant_document_references.append({
            "doc_name": doc,
            "reference": f"Document: {doc}, Reference Section: Some section details"
        })

    # Return the answer, the relevant documents, and the matching references
    return response_text, relevant_document_references





# Main Streamlit app function
def main():
    st.set_page_config(page_title="Chat With PDFs", layout="wide")
    st.header("Chat with Multiple PDFs using Google Gemini")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        with st.spinner("Generating a response..."):
            reply, docs = user_input(user_question)
            st.session_state.chat_history.append((user_question, reply, docs))

    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, (question, response, docs) in enumerate(reversed(st.session_state.chat_history)):
            st.write(f"**Q{i+1}:** {question}")
            st.write(f"**A{i+1}:** {response}")
            for doc in docs:
                st.write(f"**Relevant Document:** {doc['doc_name']}")
                st.write(f"**Reference:** {doc['reference']}")
            st.write("---")

    if not os.path.exists("faiss_index"):
        with st.spinner("Processing PDFs..."):
            pdf_texts = extract_text_from_folder(PDF_FOLDER_PATH)
            text_chunks = get_text_chunks(pdf_texts)
            get_vector_store(text_chunks)
            st.success("PDF processing completed.")

if __name__ == "__main__":
    main()
