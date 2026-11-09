
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from chunking import chunk_text


DOCUMENTS_FOLDER = Path("documents")

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

all_chunks = []
all_metadata = []
all_ids = []

document_count = 0

print("\nReading documents...")

supported_extensions = [".txt", ".md", ".pdf"]

for file_path in DOCUMENTS_FOLDER.iterdir():

    if file_path.suffix.lower() not in supported_extensions:
        continue

    # Read TXT and Markdown files
    if file_path.suffix.lower() in [".txt", ".md"]:
        text = file_path.read_text(encoding="utf-8")

    # Read PDF files
    elif file_path.suffix.lower() == ".pdf":
        pdf_reader = PdfReader(file_path)
        text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    if not text.strip():
        print(f"Skipping empty document: {file_path.name}")
        continue

    document_count += 1

    # Split text into chunks
    chunks = chunk_text(text)

    print(f"{file_path.name}: {len(chunks)} chunks")

    for index, chunk in enumerate(chunks):

        all_chunks.append(chunk)

        all_metadata.append({
            "source": file_path.name,
            "chunk_number": index + 1
        })

        all_ids.append(f"{file_path.stem}_{index + 1}")


if not all_chunks:
    print("No supported documents found!")
    exit()


print("\nCreating embeddings...")
embeddings = embedding_model.encode(all_chunks).tolist()


print("\nUpdating ChromaDB...")

client = chromadb.PersistentClient(path="chroma_db")

try:
    client.delete_collection(name="rag_documents")
except Exception:
    pass

collection = client.create_collection(name="rag_documents")

collection.add(
    ids=all_ids,
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=all_metadata
)


print("\nIngestion completed successfully!")
print("Total documents:", document_count)
print("Total chunks stored:", collection.count())
