from sentence_transformers import SentenceTransformer

import numpy as np

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

def generate_embeddings(strings: list[str]):
    print(strings)

    embeddings = model.encode(
        strings,
        convert_to_numpy=True
    )

    print(embeddings.shape)
    print(embeddings)

    return embeddings

if __name__=="__main__":
    texts=[
        "Hello how are you",
        "I like machine learning"
    ]
    embeddings = generate_embeddings(texts)