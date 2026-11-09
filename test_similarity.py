
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="rag_documents")

# 3. Ask a question interactively
question = input("\nEnter your question: ").strip()

if not question:
    print("Please enter a valid question.")
    exit()

# 4. Convert question into an embedding
question_embedding = model.encode([question]).tolist()

# 5. Retrieve the top-4 most relevant chunks
results = collection.query(
    query_embeddings=question_embedding,
    n_results=4,
    include=["documents", "metadatas", "distances"]
)

# 6. Display retrieved chunks
print("\n========== QUESTION ==========")
print(question)

print("\n========== RETRIEVED CHUNKS ==========")

for i in range(len(results["documents"][0])):
    print(f"\n--- Chunk {i + 1} ---")
    print("Source:", results["metadatas"][0][i]["source"])
    print("Distance:", round(results["distances"][0][i], 4))
    print("Content:")
    print(results["documents"][0][i])

print("\n========== RETRIEVAL COMPLETED ==========")
