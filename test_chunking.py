
from chunking import chunk_text

sample_text = """
Python is a high-level programming language.
It is widely used in artificial intelligence,
machine learning, data science and automation.
""" * 100

chunks = chunk_text(sample_text)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk[:150])
