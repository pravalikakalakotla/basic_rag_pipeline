# Basic RAG Pipeline

A beginner-friendly Retrieval-Augmented Generation (RAG) system that answers questions using information stored in personal documents. It combines semantic search with a Large Language Model (LLM) to generate context-based answers.

## Project Overview

This project implements a basic RAG pipeline using Python, Sentence Transformers, ChromaDB, and Groq.

The system reads documents, splits them into smaller overlapping chunks, converts the chunks into vector embeddings, and stores them in ChromaDB.

When a user asks a question, the system retrieves the most relevant chunks and passes them as context to the Groq LLM to generate an answer.

## Technologies Used

- Python
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
│   ├── customer_churn.txt
│   ├── federated_learning.txt
│   ├── internship_notes.txt
│   ├── java_notes.txt
│   └── python_notes.txt
│
├── chroma_db/          # Created locally during ingestion
├── venv/               # Local Python virtual environment
├── .env                # Stores the Groq API key
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

**Note:** The `venv/`, `.env`, and `chroma_db/` folders or files are created and used locally. They are excluded from the public GitHub repository through `.gitignore`.

## How It Works

1. **Document Collection:** Collects text documents from the documents folder.
2. **Text Chunking:** Splits documents into smaller chunks using a chunk size of 400 tokens and an overlap of 50 tokens.
3. **Embedding Generation:** Uses the all-MiniLM-L6-v2 model to convert text chunks into vector embeddings.
4. **Vector Storage:** Stores embeddings and their corresponding text chunks in a local ChromaDB database.
5. **Question Embedding:** Converts the user's question into a vector embedding.
6. **Retrieval:** Retrieves the top 4 most similar chunks from ChromaDB.
7. **Relevance Filtering:** Uses a distance threshold to filter out less relevant chunks.
8. **Answer Generation:** Sends the relevant context and user question to the Groq LLM to generate an answer.

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

Activate it on Windows:

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

Replace `your_groq_api_key` with your own Groq API key.

**Important:** Never upload your `.env` file or expose your API key publicly.

### 5. Run Document Ingestion

```bash
python ingest.py
```

This reads the documents, generates embeddings, and stores them in ChromaDB.

### 6. Run the RAG Pipeline

```bash
python rag_pipeline.py
```

Enter your questions in the terminal and receive answers based on the stored documents.

Type `exit` to stop the application.

## Example Questions

- What is Federated Learning and how does it work?
- Explain my Customer Churn Prediction project.
- What is quantum computing?

If relevant information is not available in the stored documents, the system returns:

`I don't have enough information in the documents.`

## Testing

The project includes separate testing scripts:

- `test_chunking.py` – Tests the document chunking process.
- `test_retrieval.py` – Checks document retrieval from ChromaDB.
- `test_similarity.py` – Tests semantic similarity and relevance ranking.

## Key Features

- Document-based question answering.
- Semantic search using vector embeddings.
- Local vector storage with ChromaDB.
- Context-based answer generation using Groq.
- Retrieval of the top 4 relevant chunks.
- Distance-based relevance filtering.
- Displays retrieved chunks with source information.
- Handles questions when relevant information is unavailable.

## Future Improvements

- Support for PDF and DOCX document formats.
- Improved chunking strategies.
- Conversation memory for follow-up questions.
- Agentic RAG with query planning and tool usage.
- Web-based user interface.

## Author

**Pravalika Kalakotla**

GitHub: https://github.com/pravalikakalakotla

## License

This project is open-source and available for learning and educational purposes.
