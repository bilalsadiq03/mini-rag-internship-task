import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ingest import chunk_documents, load_documents


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def create_embeddings(chunks):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    return embeddings, model


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index



if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)

    embeddings, model = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    print(f"Total chunks indexed: {index.ntotal}")

    faiss.write_index(index, "faiss_index.bin")
    np.save("embeddings.npy", embeddings)
    np.save("chunks_metadata.npy", chunks, allow_pickle=True)

    print("FAISS index and embeddings saved.")

