
import chromadb

# Connect to existing ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

# Access the stored collection
collection = client.get_collection(name="rag_documents")

# Display total stored chunks
print("Total chunks in database:", collection.count())

# Retrieve all stored documents
results = collection.get(
    include=["documents", "metadatas"]
)

# Display stored information
for i in range(len(results["ids"])):
    print("\n----------------------------")
    print("ID:", results["ids"][i])
    print("Source:", results["metadatas"][i]["source"])
    print("Chunk:", results["metadatas"][i]["chunk_number"])
    print("Text:", results["documents"][i][:150])
