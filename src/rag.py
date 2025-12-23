import subprocess
from retriever import Retriever


SYSTEM_PROMPT = """
You are an AI assistant for a construction marketplace.
You must answer the user question using ONLY the provided context.
If the answer is not contained in the context, reply with:
"I don't know based on the provided documents."

Do NOT use prior knowledge.
Do NOT make assumptions.
"""


def generate_answer(query, retrived_chuks):
    context = "\n\n".join(
        [f"Source: {chunk['source']}\n{chunk['content']}" for chunk in retrived_chuks]
    )
    prompt = f""" {SYSTEM_PROMPT} Context: {context} Question: {query} Answer:"""

    response = subprocess.run(
        ["ollama", "run", "phi"],
        input=prompt,
        text=True,
        capture_output=True,
        shell=True,
        encoding="utf-8",
        errors="ignore"
    )
    return response.stdout.strip()



if __name__ == "__main__":
    retriever = Retriever()

    query = input("\nEnter your question: ")
    retrieved_chunks = retriever.retrieve(query)

    # print("/nRetrieved Chunks:")
    # for i, chunk in enumerate(retrieved_chunks, 1):
    #     print(f"[{i}] Source: {chunk['source']} | Score: {chunk['score']:.4f}")
    #     print(chunk["content"])
    #     print("-" * 80)

    answer = generate_answer(query, retrieved_chunks)

    print("\nANSWER:\n")
    print(answer)

