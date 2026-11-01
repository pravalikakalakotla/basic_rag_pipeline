
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from chunking import chunk_text

# 1. Define document folder
DOCUMENTS_FOLDER = Path("documents")

# 2. Load the embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Read documents and create chunks
all_chunks = []
all_metadata = []
all_ids = []

print("\nReading documents...")

for file_path in DOCUMENTS_FOLDER.glob("*.txt"):

    text = file_path.read_text(encoding="utf-8")

    chunks = chunk_text(text)

    print(f"{file_path.name}: {len(chunks)} chunks")

    for index, chunk in enumerate(chunks):
        all_chunks.append(chunk)

        all_metadata.append({
            "source": file_path.name,
            "chunk_number": index + 1
        })

        all_ids.append(f"{file_path.stem}_{index + 1}")

# 4. Generate embeddings
print("\nGenerating embeddings...")

embeddings = embedding_model.encode(all_chunks).tolist()

# 5. Connect to ChromaDB
print("\nConnecting to ChromaDB...")

client = chromadb.PersistentClient(path="chroma_db")

# Recreate collection to avoid duplicate data on reruns
try:
    client.delete_collection(name="rag_documents")
except Exception:
    pass

collection = client.create_collection(name="rag_documents")

# 6. Store chunks, embeddings and metadata
collection.add(
    ids=all_ids,
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=all_metadata
)

print("\nIngestion completed successfully!")
print("Total documents:", len(list(DOCUMENTS_FOLDER.glob("*.txt"))))
print("Total chunks stored:", collection.count())
