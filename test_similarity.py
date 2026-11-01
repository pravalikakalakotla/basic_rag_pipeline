
import chromadb
from sentence_transformers import SentenceTransformer

# Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="rag_documents")

# Ask a question
question = "What is Federated Learning?"

# Convert question into an embedding
question_embedding = model.encode([question]).tolist()

# Search for the 3 most relevant chunks
results = collection.query(
    query_embeddings=question_embedding,
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

# Display retrieved results
print("\nQuestion:", question)
print("\nRetrieved relevant information:")

for i in range(len(results["documents"][0])):
    print("\n-------------------------")
    print("Source:", results["metadatas"][0][i]["source"])
    print("Distance:", results["distances"][0][i])
    print("Content:")
    print(results["documents"][0][i])
