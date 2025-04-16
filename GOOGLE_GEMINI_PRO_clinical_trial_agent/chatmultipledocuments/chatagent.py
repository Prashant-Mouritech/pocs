import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Load environment variables for the Google API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Path to your PDF file
PDF_PATH = r"DNDi-Clinical-Trial-Protocol-BENDITA-V5.pdf"  # Update this path accordingly

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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Free embedding model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save FAISS index locally

# Function to set up the conversational chain with Google Gemini
def get_conversational_chain():
    # Custom prompt template for QA chain
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    # Set up the Gemini model for chat
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Load a QA chain with the Gemini model and custom prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input and generate the response
def user_input(user_question):
    # Load embeddings and FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # FAISS warning handling
    docs = vector_store.similarity_search(user_question)  # Perform a similarity search on the user's question
    
    # Get the conversational chain (QA system)
    chain = get_conversational_chain()

    # Generate a response using the chain and relevant documents
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response["output_text"]  # Return the response text

# Main Streamlit app function with chat history support
def main():
    st.set_page_config(page_title="Chat With PDF", layout="wide")
    st.header("Chat with PDF using Google Gemini")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # List to store question-response pairs

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []  # Reset chat history
        st.success("Chat history cleared!")

    # User input section for questions
    user_question = st.text_input("Ask a Question from the PDF Files")

    # If the user provides a question, generate a response
    if user_question:
        with st.spinner("Generating a response..."):
            reply = user_input(user_question)  # Generate response from user's question
            # Save the question and reply to chat history
            st.session_state.chat_history.append((user_question, reply))

    # Display the chat history in reverse order (most recent first)
    if st.session_state.chat_history:
        st.subheader("Chat History")
        # Reverse the order to show the most recent question-response at the top
        for i, (question, response) in enumerate(reversed(st.session_state.chat_history)):
            st.write(f"**Q{i+1}:** {question}")
            st.write(f"**A{i+1}:** {response}")
            st.write("---")  # Divider for better readability

    # Backend processing for PDF and FAISS index creation (only once)
    if not os.path.exists("faiss_index"):  # Check if the FAISS index has already been created
        with st.spinner("Processing PDF..."):
            raw_text = get_pdf_text(PDF_PATH)  # Extract text from PDF
            text_chunks = get_text_chunks(raw_text)  # Split text into chunks
            get_vector_store(text_chunks)  # Create and save FAISS index
            st.success("PDF processing completed.")

if __name__ == "__main__":
    main()
