
# Agentic RAG Pipeline Architecture

This document explains the architecture and flow of the **Agentic Retrieval-Augmented Generation (RAG)** pipeline implemented in the provided code. The system combines **retrieval**, **reranking**, **context composition**, and **LLM synthesis** into a modular and explainable agentic workflow.

---

## Overview

The Agentic RAG pipeline integrates multiple components:

1. **Document Ingestion & Chunking** – Loads PDF files, splits them into semantically coherent chunks.
2. **Multi-Retrieval System** – Combines **dense (FAISS)** and **sparse (BM25)** retrievers.
3. **Cross-Encoder Reranking** – Uses a transformer model to score and reorder retrieved chunks.
4. **Query Planning & Intent Routing** – Decomposes and rewrites queries based on domain-specific rules.
5. **Context Composition** – Merges and summarizes the most relevant chunks under a token budget.
6. **Self-Reflection Module** – Ensures semantic overlap between question and retrieved context.
7. **Answer Synthesis (LLM)** – Generates final responses using a large reasoning model (Flan-T5-XL).
8. **Memory System** – Stores query-response pairs in a JSON file for episodic recall.

---

## System Components

### 1. **PDF Loader and Chunking**

* **Modules:** `PyPDFLoader`, `RecursiveCharacterTextSplitter`
* **Function:** Converts PDF pages into text chunks with metadata (page numbers, offsets).
* **Chunk Parameters:**

  * `chunk_size = 700`
  * `chunk_overlap = 120`
* **Output:** List of document chunks stored as LangChain Document objects.

### 2. **Embeddings and Vector Store (Dense Retrieval)**

* **Model:** `sentence-transformers/all-mpnet-base-v2`
* **Store:** `FAISS`
* **Purpose:** Computes dense embeddings for each chunk and performs cosine similarity search.
* **Returns:** Top-K semantically relevant chunks.

### 3. **BM25 Sparse Retriever**

* **Library:** `rank_bm25`
* **Purpose:** Keyword-based retrieval for complementing semantic retrieval.
* **Top-K:** 10 results.

### 4. **Cross-Encoder Reranker**

* **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Purpose:** Scores each retrieved chunk against the query using a cross-encoder architecture.
* **Computation:**

  * Tokenizes query-context pairs.
  * Computes relevance logits.
  * Sorts results by descending scores.

### 5. **Query Planning & Decomposition**

* **Functions:** `rewrite()`, `route_intent()`, `decompose()`
* **Purpose:** Enhances retrieval performance by rewriting domain-specific queries.
* **Logic:**

  * Expands astrophysical terminology (e.g., *MESA → Modules for Experiments in Stellar Astrophysics*).
  * Detects query type: `definition`, `comparison`, `howto`, or `generic`.
  * Splits multi-part questions into atomic sub-queries.

### 6. **Tools and Agent Definition**

Each step in the workflow is wrapped as a modular **Tool** subclass compatible with the `smolagents` framework:

| Tool                  | Description                                               |
| --------------------- | --------------------------------------------------------- |
| `QueryPlanTool`       | Rewrites and classifies queries.                          |
| `VectorRetrieverTool` | Retrieves semantically similar chunks using FAISS.        |
| `BM25RetrieverTool`   | Retrieves keyword matches using BM25.                     |
| `RerankTool`          | Reorders results with cross-encoder scores.               |
| `ComposeContextTool`  | Merges top-ranked chunks into a context window.           |
| `SelfReflectTool`     | Evaluates context relevance and triggers retry if needed. |

### 7. **Agent Workflow**

* **Framework:** `smolagents.CodeAgent`
* **Configuration:**

  * `max_steps = 6`
  * `verbosity_level = 2`
* **Flow:**

  1. Query → Query Planner (rewrite, decompose, classify)
  2. Dense Retrieval (FAISS)
  3. Sparse Retrieval (BM25)
  4. Merge & Rerank
  5. Context Composition
  6. Self-Reflection
  7. Optional Retry if overlap < 0.5

### 8. **Episodic Memory System**

* **File:** `agent_memory.json`
* **Structure:**

  ```json
  {
    "episodes": [
      {
        "ts": 1731430000.0,
        "query": "...",
        "answer": "...",
        "sources": [{"page": 3, "preview": "..."}]
      }
    ]
  }
  ```
* **Functions:**

  * `remember()` → Appends query, answer, and sources.
  * `episodic_recall()` → Finds similar past queries using Jaccard similarity.

### 9. **LLM Answer Generation**

* **Model:** `google/flan-t5-xl`
* **Pipeline:** Hugging Face `text2text-generation`
* **Input:** Agent-generated context + original query.
* **Output:** Coherent, factual answer (≤ 800 tokens).

---

## Agentic RAG Execution Flow

1. **User Query → Query Planner**

   * Expands domain terms and classifies intent.
2. **Retrieval Phase**

   * Performs FAISS dense retrieval and BM25 sparse retrieval.
   * Merges results.
3. **Reranking Phase**

   * Scores all retrieved chunks using the cross-encoder.
4. **Context Composition**

   * Selects top-ranked chunks under a 1700-token limit.
5. **Self-Reflection**

   * Checks query-context overlap. If low, retries retrieval.
6. **LLM Synthesis**

   * Flan-T5 synthesizes the final answer.
7. **Memory Logging**

   * Query, answer, and source metadata stored in `agent_memory.json`.

---

## Example Flow (MESA Query)

**Input:**

> "Summarize how MESA’s modular design allows integration of new physical modules and propose how an agent could extend it to model accreting white dwarfs."

**Pipeline Steps:**

1. Query Planner expands to include synonyms like *rotational mixing*, *degenerate star*, etc.
2. FAISS + BM25 retrieve relevant text chunks.
3. Cross-encoder reranks based on semantic relevance.
4. Context composer builds a concise, token-bounded input.
5. Self-reflection validates coverage → triggers optional retry.
6. Final synthesis by Flan-T5-XL → generates a structured, domain-aware response.

---

## Key Features

* Dual retrieval system (Dense + Sparse)
* Cross-encoder reranking for precision
* Domain-aware query rewriting for astrophysics/MESA
* Episodic long-term memory for contextual recall
* Reflexive loop for improved retrieval reliability
* Modular design with `Tool` abstraction
* End-to-end reasoning with Flan-T5 synthesis

---

## Summary Table

| Layer         | Module                                 | Purpose                   |
| ------------- | -------------------------------------- | ------------------------- |
| Input         | `files.upload()`                       | PDF ingestion             |
| Preprocessing | `RecursiveCharacterTextSplitter`       | Chunking                  |
| Retrieval     | `FAISS`, `BM25Retriever`               | Dense + Sparse search     |
| Reranking     | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision scoring         |
| Context       | `ComposeContextTool`                   | Merge within token budget |
| Reflection    | `SelfReflectTool`                      | Quality control           |
| Synthesis     | `google/flan-t5-xl`                    | Final answer generation   |
| Memory        | JSON episodic store                    | Query recall              |

---

**Author:** Rahul Thakur
**Purpose:** Explainable documentation for the Agentic RAG pipeline built using LangChain, SmolAgents, HuggingFace, and FAISS.
