# Basic RAG Pipeline

## Project Overview

This project implements a Basic Retrieval-Augmented Generation (RAG) pipeline using Python, Sentence Transformers, ChromaDB, and Groq.

The system allows users to ask questions about their own documents. It retrieves relevant information from the documents and uses a Large Language Model (LLM) to generate context-based answers.

## Features

- Reads text documents from a local folder.
- Splits documents into overlapping text chunks.
- Generates embeddings using Sentence Transformers.
- Stores embeddings in ChromaDB.
- Retrieves relevant document chunks using semantic similarity.
- Generates answers using the Groq API.
- Avoids answering questions when the required information is not available in the documents.

## Technologies Used

- Python 3.11
- Sentence Transformers
- ChromaDB
- Groq API
- tiktoken
- python-dotenv

## Project Structure

```text
basic-rag-pipeline/
│
├── documents/
│   ├── customer_churn.txt
│   ├── federated_learning.txt
│   ├── python_notes.txt
│   ├── java_notes.txt
│   └── internship_notes.txt
│
├── chroma_db/
├── venv/
├── .env
├── .gitignore
├── chunking.py
├── ingest.py
├── rag_pipeline.py
├── test_chunking.py
├── test_retrieval.py
├── test_similarity.py
├── README.md
└── requirements.txt
```

## RAG Pipeline Architecture

1. Document Loading – Reads text documents.
2. Text Chunking – Splits documents into chunks of 400 tokens with 50-token overlap.
3. Embedding Generation – Converts text chunks into numerical vectors using `all-MiniLM-L6-v2`.
4. Vector Storage – Stores document chunks, embeddings, and metadata in ChromaDB.
5. Similarity Search – Retrieves relevant chunks for a user question.
6. Answer Generation – Sends the retrieved context to Groq to generate an answer.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd basic-rag-pipeline
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Groq API key

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_groq_api_key
```

Never upload your API key to GitHub.

### 6. Add documents

Place your `.txt` documents inside the `documents` folder.

### 7. Run document ingestion

```bash
python ingest.py
```

### 8. Start the RAG pipeline

```bash
python rag_pipeline.py
```

Ask questions through the terminal. Type `exit` to close the application.

## Testing

The pipeline was tested using questions about:

- Federated Learning for Privacy Preserving Healthcare.
- Customer Churn Prediction.
- Python and Java programming.
- RAG concepts.

An out-of-scope question about quantum computing was also tested. The system correctly indicated that the information was unavailable in its documents.

## Future Improvements

- Support PDF and other document formats.
- Improve chunking and retrieval strategies.
- Build a Streamlit-based user interface.
- Add conversation history.
- Improve source citations and retrieval accuracy.

## Author

Developed as a hands-on project to understand and implement Retrieval-Augmented Generation.
