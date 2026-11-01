
import tiktoken

# Chunk configuration
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Load tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")


def chunk_text(text):
    # Convert text into tokens
    tokens = tokenizer.encode(text)

    chunks = []

    # Calculate step size
    step = CHUNK_SIZE - CHUNK_OVERLAP

    # Split tokens into overlapping chunks
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + CHUNK_SIZE]

        # Convert tokens back into text
        chunk = tokenizer.decode(chunk_tokens)

        if chunk.strip():
            chunks.append(chunk)

    return chunks
