# Mini RAG System For Indecimal (Internship Assignment)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for an AI assistant designed to answer user questions **strictly using internal construction-related documents** (policies, FAQs, specifications).

The system retrieves relevant document chunks using semantic search and generates grounded answers using a Large Language Model.

--

## Objective

- Retrieve relevant information from provided internal documents
- Generate answers grounded strictly in retrieved content
- Ensure transparency, explainability, and minimal hallucination
- Demonstrate understanding of embeddings, vector search, and RAG systems

---


##  System Architecture

User Query  
↓  
Query Embedding  
↓  
FAISS Vector Search (Top-K Chunks)  
↓  
Retrieved Context  
↓  
Local LLM (Ollama – Phi-2)  
↓  
Grounded Answer

---

##  Document Processing

- Documents are loaded from Markdown files
- Chunked using `RecursiveCharacterTextSplitter`
- Chunk size: ~600 tokens with overlap
- Each chunk retains source metadata for transparency

---

##  Embeddings & Vector Search

- **Embedding Model:** `all-MiniLM-L6-v2` (Sentence Transformers)
  - Lightweight, fast, and effective for semantic similarity
- **Vector Store:** FAISS (local)
  - Cosine similarity via normalized embeddings
- **Top-K Retrieval:** 3 chunks per query

---

##  Answer Generation (Grounded RAG)

- A local open-source LLM is used via **Ollama**
- Model: **phi-2 (~2.7B parameters)**
- The LLM is explicitly instructed to:
  - Answer only using retrieved context
  - Say *"I don't know based on the provided documents"* if information is missing
- Retrieved context is always displayed before the final answer

---


##  Evaluation & Quality Analysis

### Test Setup
- 10 evaluation questions derived directly from the documents and 5 questions out of documents
- Manual qualitative analysis performed

### Observations
- Retrieval results were mostly relevant and aligned with queries
- Answers were consistently grounded in retrieved content
- No major hallucinations observed
- Local LLM produced concise and accurate responses

### Limitations
- Small document set limits answer depth
- No reranking or hybrid (BM25 + vector) search
- Local LLM fluency slightly lower than API-based models


## How to Run the Project

### 1. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate   
pip install -r requirements.txt
```

### 2. Build Embeddings & Index
```bash
python src/embed.py
```

### 3. Run the RAG System
```bash
python src/rag.py
```


### Author
Bilal Sadiq