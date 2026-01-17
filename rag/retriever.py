import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(chunks):
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, chunks

def retrieve_relevant_chunks(query, index, chunks, k=3):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding), k)

    return [chunks[i] for i in indices[0]]
