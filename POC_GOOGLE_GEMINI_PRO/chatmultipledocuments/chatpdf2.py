import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Chunking the text
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Vectorization
import google.generativeai as genai
from langchain.vectorstores import FAISS  # Vector Store
from langchain_google_genai import ChatGoogleGenerativeAI  # For chat
from langchain.chains.question_answering import load_qa_chain  # Do the chat
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Specify your PDF file path in the backend
PDF_PATH = r"C:\Users\prashantvi\Desktop\POC_GOOGLE_GEMINI_PRO\chatmultipledocuments\DNDi-Clinical-Trial-Protocol-BENDITA-V5.pdf"  # Update this to the actual path


# Function to extract text from PDF
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to store the embeddings in a FAISS index
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Using this because it is free
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Function to load the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Function to handle user input and generate a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # FAISS warning on loading data.
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    # Generate a response using the conversational chain
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]


# Main function for Streamlit app
def main():
    st.set_page_config("Chat With PDF")
    st.header("Chat with PDF using Gemini")

    # Display the user input (question) box in the frontend
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        with st.spinner("Generating a response..."):
            reply = user_input(user_question)  # Generate response based on the user's question
            st.write("Reply: ", reply)

    # Backend processing of PDF file
    if not os.path.exists("faiss_index"):  # Check if the FAISS index has already been created
        with st.spinner("Processing PDF..."):
            raw_text = get_pdf_text(PDF_PATH)  # Read the PDF text from the backend
            text_chunks = get_text_chunks(raw_text)  # Split the text into chunks
            get_vector_store(text_chunks)  # Create and save FAISS index
            st.success("PDF processing completed.")


if __name__ == "__main__":
    main()
