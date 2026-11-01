
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


def answer_question(question):

    # 5. Convert question into embedding
    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    # 6. Retrieve relevant chunks
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=4,
        include=["documents", "metadatas"]
    )

    # 7. Prepare context
    context_parts = []

    for document, metadata in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        context_parts.append(
            f"Source: {metadata['source']}\n{document}"
        )

    context = "\n\n".join(context_parts)

    # 8. Create prompt
    prompt = f"""
    Answer the question using only the information
    provided in the context below.

    Give a clear and complete answer.
    If the answer is not available in the context,
    say: "I don't have enough information in the documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # 9. Generate answer using Groq
    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful document-based RAG assistant."
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


# 10. Ask questions continuously
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
            answer = answer_question(question)
            print("\nAI Answer:")
            print(answer)
            print("\n" + "=" * 50 + "\n")

        except Exception as error:
            print("\nError:", error)
