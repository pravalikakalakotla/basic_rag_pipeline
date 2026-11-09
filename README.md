# Basic RAG Pipeline

A beginner-friendly Retrieval-Augmented Generation (RAG) system that answers questions using information stored in personal documents. It combines semantic search with a Large Language Model (LLM) to generate context-aware answers based on retrieved information.

## Project Overview

This project implements a basic RAG pipeline using Python, Sentence Transformers, ChromaDB, and Groq.

The system reads documents in TXT, Markdown, and PDF formats, splits the text into smaller overlapping chunks, converts the chunks into vector embeddings, and stores them in a local ChromaDB database.

When a user asks a question, the system converts the question into an embedding, retrieves the top 4 most relevant chunks, filters them using a distance threshold, and passes the relevant context to the Groq LLM to generate an answer.

If the required information is not available in the documents, the system returns an appropriate response instead of generating an unsupported answer.

## Technologies Used

- Python 3.11
- Sentence Transformers
- ChromaDB
- Groq API
- all-MiniLM-L6-v2 (Embedding Model)
- Tiktoken
- PyPDF
- Python-dotenv

## Project Structure

```text
basic_rag_pipeline/
│
├── documents/
│   ├── customer_churn_readme.md
│   ├── internship_notes.txt
│   ├── java_notes.txt
│   ├── python_notes.txt
│   └── REPORT_SEMINAR_B24CS119.pdf
│
├── chroma/               # Local ChromaDB storage
├── venv/                 # Python virtual environment
├── .env                  # Groq API key
├── .gitignore
├── chunking.py
├── ingest.py
├── rag_pipeline.py
├── requirements.txt
├── test_chunking.py
├── test_retrieval.py
├── test_similarity.py
└── README.md
```

**Note:** The `venv/`, `.env`, and local ChromaDB storage folders are used locally and excluded from the public GitHub repository through `.gitignore`.

## How It Works

1. **Document Collection:** Reads personal documents from the `documents/` folder in TXT, Markdown, and PDF formats.
2. **Text Chunking:** Splits document text into smaller chunks using sentence-aware chunking, with an approximate chunk size of 400 tokens and an overlap of up to 50 tokens.
3. **Embedding Generation:** Uses the `all-MiniLM-L6-v2` model to convert text chunks into vector embeddings.
4. **Vector Storage:** Stores the embeddings, text content, and metadata in a local ChromaDB database.
5. **Question Embedding:** Converts the user's question into a vector embedding.
6. **Retrieval:** Retrieves the top 4 most similar chunks from ChromaDB.
7. **Relevance Filtering:** Applies a distance threshold to filter out less relevant chunks.
8. **Answer Generation:** Sends the relevant context and user question to the Groq LLM.
9. **Response:** Generates a context-based answer and displays the retrieved chunks with source information.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pravalikakalakotla/basic_rag_pipeline.git
cd basic_rag_pipeline
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API Key

Create a `.env` file in the project root directory and add:

```env
GROQ_API_KEY=your_groq_api_key
```

Replace `your_groq_api_key` with your actual Groq API key.

**Important:** Never upload your `.env` file or expose your API key publicly.

### 5. Run Document Ingestion

```bash
python ingest.py
```

This reads the documents, splits them into chunks, generates embeddings, and stores them in ChromaDB.

### 6. Run the RAG Pipeline

```bash
python rag_pipeline.py
```

Enter your questions in the terminal and receive answers based on the stored documents.

Type `exit` to stop the application.

## Example Questions

- What is Federated Learning?
- Explain my Customer Churn Prediction project.
- What is Java inheritance?
- Explain Python basics.
- What is quantum computing?

If relevant information is not available in the stored documents, the system returns:

```text
I don't have enough information in the documents.
```

## Testing

The project includes separate testing scripts:

- `test_chunking.py` – Tests the document chunking process.
- `test_retrieval.py` – Checks document retrieval from ChromaDB.
- `test_similarity.py` – Tests semantic similarity and displays retrieved chunks.

## Key Features

- Document-based question answering.
- Supports TXT, Markdown, and PDF documents.
- Sentence-aware text chunking.
- Semantic search using vector embeddings.
- Local vector storage with ChromaDB.
- Context-based answer generation using Groq.
- Retrieval of the top 4 relevant chunks.
- Distance-based relevance filtering.
- Displays retrieved chunks with source, distance, and content.
- Handles questions when relevant information is unavailable.
- Local embedding generation using Sentence Transformers.

## Future Improvements

- Support for DOCX document format.
- Improved retrieval accuracy and ranking.
- Conversation memory for follow-up questions.
- Agentic RAG with query planning and tool usage.
- Web-based user interface.
- Hybrid search combining semantic and keyword retrieval.

## Author

**Pravalika Kalakotla**

GitHub: https://github.com/pravalikakalakotla

## License

This project is open-source and available for learning and educational purposes.
