
import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key not found. Check your .env file.")

# 2. Initialize Groq
groq_client = Groq(api_key=api_key)

MODEL_NAME = "openai/gpt-oss-20b"

# 3. Load embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="rag_documents")

# 5. Distance threshold
DISTANCE_THRESHOLD = 1.2


# 6. Main RAG query function
def query(question):

    # Convert question into embedding
    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    # Retrieve top 4 relevant chunks
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=4,
        include=["documents", "metadatas", "distances"]
    )

    context_parts = []

    print("\n========== RETRIEVED CHUNKS ==========")

    # Filter and display relevant chunks
    for i, (document, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ),
        start=1
    ):

        if distance > DISTANCE_THRESHOLD:
            continue

        print(f"\nChunk {i}")
        print(f"Source: {metadata['source']}")
        print(f"Distance: {distance:.4f}")
        print(f"Content:\n{document}")
        print("-" * 60)

        context_parts.append(
            f"Source: {metadata['source']}\n{document}"
        )

    # Check whether relevant context was found
    if not context_parts:
        print("No relevant chunks found.")
        return "I don't have enough information in the documents."

    context = "\n\n".join(context_parts)

    # 7. Create prompt
    prompt = f"""
    Answer the question using only the information
    provided in the context below.

    Give a clear and complete answer.
    Do not include unrelated information.

    If the answer is not available in the context,
    say: "I don't have enough information in the documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # 8. Generate answer using Groq
    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful document-based RAG assistant. Answer only from the supplied context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=1000
    )

    return response.choices[0].message.content


# 9. Ask questions continuously
if __name__ == "__main__":

    print("\n===== BASIC RAG PIPELINE =====")
    print("Ask questions about your documents.")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            print("RAG Pipeline closed.")
            break

        if not question:
            continue

        try:
            answer = query(question)

            print("\n========== GENERATED ANSWER ==========")
            print(answer)
            print("\n" + "=" * 60 + "\n")

        except Exception as error:
            print("\nError:", error)
