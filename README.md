# Clinical Trial Chatbot with Azure OpenAI and Azure Search

This project allows users to chat with a PDF document (e.g., clinical trial protocols) using Azure OpenAI's GPT-3.5-turbo model and Azure Cognitive Search. The PDF is processed, indexed, and stored in Azure Cognitive Search, enabling semantic search based on the content. The chatbot generates responses based on user queries using contextual information from the PDF document.

## Features

- **PDF Text Extraction**: Extracts text from a PDF document using `PyPDF2`.
- **Text Chunking**: Splits large PDF content into manageable chunks using `langchain.text_splitter`.
- **Azure Search Indexing**: Creates an Azure Search index and indexes text chunks with embeddings.
- **Semantic Search**: Uses embeddings to perform semantic search and retrieve relevant chunks.
- **Chat with GPT-3.5**: Uses Azure OpenAI's GPT-3.5-turbo model to generate responses based on the search results.

## Technologies Used

- **Azure Cognitive Search**: For indexing and searching PDF content.
- **Azure OpenAI**: For generating embeddings and responses.
- **Streamlit**: For creating the web-based chat interface.
- **PyPDF2**: For extracting text from PDF files.
- **LangChain**: For splitting long text into chunks.
- **OpenAI's GPT-3.5-turbo**: For generating responses based on user queries.

## Requirements

- Python 3.7+
- `streamlit`
- `langchain`
- `azure-search-documents`
- `azure-identity`
- `openai`
- `PyPDF2`
- `python-dotenv`

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/PraneethVenkata/Internal-projects.git
cd Internal-projects
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory of your project with the following values:

```env
AZURE_SEARCH_ENDPOINT="your-azure-search-endpoint"
AZURE_SEARCH_ADMIN_KEY="your-azure-search-admin-key"
AZURE_OPENAI_ENDPOINT="your-azure-openai-endpoint"
AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
API_VERSION="your-azure-openai-api-version"  # e.g., '2023-05-15'
```

Replace the values above with your own Azure Search and OpenAI credentials.

### 3. Install Dependencies

Use `pip` to install the required Python libraries:

```bash
pip install -r requirements.txt
```

The `requirements.txt` should include:

```
streamlit
langchain
azure-search-documents
azure-identity
openai
PyPDF2
python-dotenv
```

### 4. Download the PDF

Ensure that the PDF you want to process (`DNDi-Clinical-Trial-Protocol-BENDITA-V5.pdf`) is in the specified path (or update the path in the code):

```python
PDF_PATH = r"C:\path\to\your\pdf\DNDi-Clinical-Trial-Protocol-BENDITA-V5.pdf"
```

### 5. Run the Application

Once everything is set up, you can start the Streamlit app by running:

```bash
streamlit run app.py
```

This will open a local development server where you can interact with the PDF document through a chatbot interface.

## How It Works

1. **Text Extraction**:
   - The PDF is read using `PyPDF2`, and the content is extracted from each page.
   - The raw text is processed into smaller chunks for easier indexing.

2. **Index Creation**:
   - An index is created in Azure Cognitive Search if it doesn't already exist.
   - The index includes fields for document `id`, `fileName`, `content`, and `contentEmbeddings` (which are vector embeddings).

3. **Embeddings and Indexing**:
   - Each chunk of text is embedded using the `text-embedding-ada-002` model from Azure OpenAI.
   - The embeddings are then uploaded to Azure Search, where they are stored alongside the text content.

4. **Search and Response**:
   - When the user asks a question, the question is converted into embeddings and used to search for relevant chunks in the indexed document.
   - The chatbot uses GPT-3.5 to generate a response based on the search results, which are displayed to the user.

## Components Breakdown

### `extract_text_from_pdf(pdf_path)`
Extracts text from the specified PDF file using `PyPDF2`.

### `get_text_chunks(pdf_text)`
Splits the extracted text into smaller chunks using `RecursiveCharacterTextSplitter` from LangChain.

### `create_index()`
Creates an Azure Cognitive Search index if it doesn't already exist, defining fields like `id`, `fileName`, and `content`.

### `index_documents_to_azure_search(text_chunks)`
Generates embeddings for each text chunk using Azure OpenAI and indexes them in Azure Search.

### `search_embeddings(user_question)`
Generates embeddings for the user question and uses Azure Search to find relevant document chunks.

### `get_chat_response(user_question)`
Combines the search results with a prompt for GPT-3.5-turbo to generate a response based on the context provided by the search results.

### `main()`
The Streamlit interface that allows users to interact with the application. It handles chat history and user input.

## Contributing

Feel free to fork this project, make improvements, or suggest features! If you want to contribute, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -m 'Add new feature'`)
4. Push to your branch (`git push origin feature-branch`)
5. Create a pull request

