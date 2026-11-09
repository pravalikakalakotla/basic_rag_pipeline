
import tiktoken
import re

# Chunk configuration
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Load tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")


def chunk_text(text):
    # Clean extra spaces and newlines
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Split text into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_sentences = []
    current_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_tokens = len(tokenizer.encode(sentence))

        # Handle sentences larger than the chunk size
        if sentence_tokens > CHUNK_SIZE:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0

            tokens = tokenizer.encode(sentence)

            for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_tokens = tokens[i:i + CHUNK_SIZE]
                chunk = tokenizer.decode(chunk_tokens)

                if chunk.strip():
                    chunks.append(chunk)

            continue

        # If adding the sentence exceeds chunk size
        if current_tokens + sentence_tokens > CHUNK_SIZE:

            # Save the current chunk
            if current_sentences:
                chunks.append(" ".join(current_sentences))

            # Keep approximately 50 tokens as overlap
            overlap_sentences = []
            overlap_tokens = 0

            for previous_sentence in reversed(current_sentences):
                previous_tokens = len(
                    tokenizer.encode(previous_sentence)
                )

                if overlap_tokens + previous_tokens > CHUNK_OVERLAP:
                    break

                overlap_sentences.insert(0, previous_sentence)
                overlap_tokens += previous_tokens

            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        # Add the new sentence
        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Store the final chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
