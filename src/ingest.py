import os
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_DIR = "data"


def load_documents() -> List[Dict]:
    documents = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".md"):
            file_path = os.path.join(DATA_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

                documents.append({
                    "source": filename, 
                    "text": content
                })

    return documents




def chunk_documents(documents: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
    )

    chunks = []

    for doc in documents:
        split_texts = splitter.split_text(doc["text"])
        for i, chunk in enumerate(split_texts):
            chunks.append({
                "source": doc["source"],
                "chunk_id": i,
                "content": chunk
            })

    return chunks



if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)


    print(f"Loaded {len(docs)} documents.")
    print(f"Created {len(chunks)} chunks.")

    print("Sample chunk:")
    print(chunks[0])