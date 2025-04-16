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

# Load environment variables for the Google API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
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
            text = "".join([page.extract_text() for page in pdf_reader.pages])
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

# Function to store text embeddings in a FAISS index with document information
def get_vector_store(text_chunks):
    # Initialize the Google Generative AI embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create an array to store text data and metadata
    text_data = []
    metadata = []

    for i, (filename, chunk) in enumerate(text_chunks):
        metadata.append({
            "id": str(i),
            "fileName": filename,  # Store the file name
            "content": chunk
        })
        text_data.append(chunk)
    # print(embeddings)

    # Create the FAISS vector store with embeddings
    vector_store = FAISS.from_texts(text_data, embedding=embeddings, metadatas=metadata)
    vector_store.save_local("faiss_index")  # Save FAISS index locally

    print("FAISS index with document metadata created and saved locally.")

# Function to create a summarization chain
def get_summarization_chain(llm):
    # Define how each document is formatted
    document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )
    document_variable_name = "context"

    # Create a summarization prompt
    summarization_prompt = ChatPromptTemplate.from_template(
        "Summarize this content: {context}"
    )

    # Create LLMChain and StuffDocumentsChain for summarization
    llm_chain = LLMChain(llm=llm, prompt=summarization_prompt)
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
    )

    return chain

# Function to set up the conversational chain with Google Gemini
# Function to set up the conversational chain with Google Gemini
def get_conversational_chain():
    # Custom prompt template for QA chain
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """

    # Set up the Gemini model for chat
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Load a QA chain with the Gemini model and custom prompt
    chain = LLMChain(llm=model, prompt=prompt)

    return chain

# Function to handle user input and generate the response
def user_input(user_question):
    # Load embeddings and FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)  # Perform a similarity search on the user's question

    # Extract context from the documents
    context = "\n\n".join([doc.page_content for doc in docs])

    # Get the conversational chain (QA system)
    chain = get_conversational_chain()

    # Generate a response using the chain and relevant documents
    response = chain({"context": context, "question": user_question})
    print("This is response",response)

    # Verify the response format and extract the output
    if "output_text" in response:
        response_text = response["output_text"]
    elif isinstance(response, dict):
        response_text = next(iter(response.values()), "No response generated.")
    else:
        response_text = "No response generated."

    # Include document names in the response
    relevant_docs = set(doc.metadata["fileName"] for doc in docs)

    return response_text, relevant_docs


# Main Streamlit app function with chat history support
def main():
    st.set_page_config(page_title="Chat With PDFs", layout="wide")
    st.header("Chat with Multiple PDFs using Google Gemini")

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
            reply, docs = user_input(user_question)  # Generate response from user's question
            # Save the question and reply to chat history
            st.session_state.chat_history.append((user_question, reply, docs))

    # Display the chat history in reverse order (most recent first)
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, (question, response, docs) in enumerate(reversed(st.session_state.chat_history)):
            st.write(f"**Q{i+1}:** {question}")
            st.write(f"**A{i+1}:** {response}")
            st.write(f"**Relevant Documents:** {', '.join(docs)}")
            st.write("---")  # Divider for better readability

    # Backend processing for PDFs and FAISS index creation (only once)
    if not os.path.exists("faiss_index"):  # Check if the FAISS index has already been created
        with st.spinner("Processing PDFs..."):
            pdf_texts = extract_text_from_folder(PDF_FOLDER_PATH)  # Extract text from all PDFs
            text_chunks = get_text_chunks(pdf_texts)  # Split text into chunks
            get_vector_store(text_chunks)  # Create and save FAISS index
            st.success("PDF processing completed.")

if __name__ == "__main__":
    main()
