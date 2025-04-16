# Install necessary libraries
!pip install -q pypdf
!pip install -q python-dotenv
!pip install -q transformers
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
!pip install -q llama-index

# Required imports and configurations
import logging
import sys
import torch
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.embeddings import LangchainEmbedding
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load documents from directory
documents = SimpleDirectoryReader("/content/data/").load_data()

# Setup the LlamaCPP model (Mistral-7B-Instruct)
llm = LlamaCPP(
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,  # Keep context window at a safe level
    model_kwargs={"n_gpu_layers": -1},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

# Set up the embeddings model for document vectorization
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="thenlper/gte-large")
)

# Configure the service context with chunk size and LLM
service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)

# Create the index using the documents and service context
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Set up a conversation history to store previous interactions
conversation_history = []

# Function to manage query and keep the conversation context
def get_response(query, query_engine, history):
    # Concatenate history to maintain context
    full_context = " ".join(history) + f"User: {query}\n"
    response = query_engine.query(full_context)  # Pass full context to the query engine
    history.append(f"Assistant: {response}\n")  # Append current response to history
    return response

# Initialize the query engine
query_engine = index.as_query_engine()

# Interact with the model and remember context
while True:
    query = input("Ask a question: ")
    
    if query.lower() in ["exit", "quit"]:
        break  # Exit loop on exit command
    
    response = get_response(query, query_engine, conversation_history)  # Pass conversation history
    print(f"Assistant: {response}")
