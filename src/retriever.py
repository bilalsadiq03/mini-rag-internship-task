import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3



class Retriever:
    def __init__(self):

        self.index = faiss.read_index("faiss_index.bin")
        
        self.chunks = np.load(
            "chunks_metadata.npy", allow_pickle=True
        ).tolist()

        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def retrieve(self, query:str, top_k: int = TOP_K):
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append({
                "content": self.chunks[idx]["content"],
                "source": self.chunks[idx]["source"],
                "score": float(score)
            })

        return results


if __name__ == "__main__":
    retriever = Retriever()

    # query = "How does Indecimal ensure transparency in construction projects?"
    query = input("Enter your query: ")
    results = retriever.retrieve(query)

    print("\nUSER QUERY:")

    print(query)

    # print("\nRETRIEVED CONTEXT:\n")

    # for i, result in enumerate(results, 1):
    #     print(f"[{i}] Source: {result['source']} | Score: {result['score']:.4f}")
    #     print(result["content"])
    #     print("-" * 80)