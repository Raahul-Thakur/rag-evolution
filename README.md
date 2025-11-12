# RAG Evolution

**Exploring the Evolution of Retrieval-Augmented Generation Architectures**

This project investigates the performance and behavior of three generations of RAG (Retrieval-Augmented Generation) systems — from the simplest vanilla setup to a fully **agentic** pipeline that reasons and retrieves adaptively. The goal is to analyze how retrieval strategy, context reasoning, and agent orchestration affect answer quality and coherence.

---

## Project Overview

| Architecture            | Description                                                                                                                                                     |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Naive / Vanilla RAG** | A baseline implementation using a static retriever and a simple LLM query pipeline. No reasoning, no adaptive retrieval.                                        |
| **Advanced RAG**        | Enhanced with multi-stage retrieval, reranking, and summarization layers for higher-quality context synthesis.                                                  |
| **Agentic RAG**         | Incorporates autonomous agents that reason, plan, and call tools (retrievers, summarizers, rerankers) dynamically — inspired by LLM-driven reasoning workflows. |

Each architecture is implemented separately to isolate differences in performance and reasoning style.

---

## Repository Structure

```
rag-evolution/
│
├── naive_rag/
│   ├── naive_rag.py
│   └── naive_rag.md
│
├── advanced_rag/
│   ├── advanced_rag.py
│   └── advanced_rag.md
│
├── agentic_rag/
│   ├── agentic_rag.py
│   └── agentic_rag.md
│
└── results/
    ├── rag_analysis.md
    ├── summarizer_analysis.md
    └── queries_and_answers.md
```

---

## Methodology

Each RAG variant was tested using a consistent set of queries, datasets, and evaluation metrics. The experiment focuses on:

* **Retrieval Quality:** Relevance and diversity of retrieved documents.
* **Context Utilization:** How effectively the model integrates retrieved context into responses.
* **Response Quality:** Accuracy, completeness, and coherence.
* **Summarization Performance:** Evaluated independently for advanced and agentic RAGs.

---

## RAG Variants in Detail

### 1. Naive RAG

* Simple FAISS/BM25 retriever
* Static top-k retrieval
* Single-pass LLM query with no feedback loop
* Serves as baseline for comparison

### 2. Advanced RAG

* Combines dense + lexical retrieval
* Uses reranking (e.g., cross-encoder or similarity scoring)
* Includes summarization and context synthesis layers
* Improved chunking and document preprocessing

### 3. Agentic RAG

* Implements reasoning-based retrieval
* Uses an LLM controller to guide multi-step tool calls
* Can dynamically decide whether to retrieve, summarize, or re-query
* Demonstrates autonomous orchestration of sub-agents

---

## Tech Stack

* **Python 3.10+**
* **LangChain / Smolagents**
* **Hugging Face Transformers**
* **FAISS / BM25Retriever**
* **Sentence Transformers**
* **OpenAI / Local LLMs**
* **pypdf, numpy, pandas, torch**

---
