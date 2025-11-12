
# Real RAG v2 — Architecture & Code

This document explains the **Real RAG v2** pipeline in simple terms and includes the **full Colab-ready code**. All text is Markdown, and all code blocks are properly closed.

---

## 1) Environment Setup

```bash
# Install dependencies (Colab-friendly)
!pip install -qU \
  langchain langchain-community langchain-huggingface \
  sentence-transformers faiss-cpu pypdf rank_bm25 \
  transformers accelerate langchain-text-splitters
```

**What this installs**

* **LangChain** building blocks (loaders, splitters, vector stores)
* **Sentence-Transformers** for embeddings (`all-mpnet-base-v2`)
* **FAISS** for dense vector search
* **rank_bm25** for keyword (sparse) search
* **Transformers** for rerankers and LLMs

---

## 2) High-Level Flow

```
PDF → Load → Chunk → (Embeddings + BM25) → Merge → Cross-Encoder Rerank →
Order + Summarize (semantic compression) → Domain-tuned Prompt → LLM Answer
```

**Key ideas**

* **Hybrid retrieval**: Combine **dense** (semantic) and **sparse** (keyword) signals
* **Reranking**: A cross-encoder scores query–passage pairs precisely
* **Semantic compression**: Keep the *most* relevant technical content
* **Domain prompt**: Steer the LLM to answer *only* from provided context

---

## 3) Module Overview

* **Loader & Splitter**: `PyPDFLoader` + `RecursiveCharacterTextSplitter`
* **Dense retriever**: `FAISS` with `all-mpnet-base-v2` embeddings
* **Sparse retriever**: `BM25Retriever`
* **Reranker**: `cross-encoder/ms-marco-MiniLM-L-12-v2`
* **Summarizer**: `flan-t5-large` (semantic compression)
* **Answer LLM**: `flan-t5-xl` (final synthesis)

---

## 4) Design Improvements (vs. Naive RAG)

* **Weighted fusion** of BM25 + FAISS (biases toward semantic recall for technical text)
* **Upgraded reranker** (L-12) for finer relevance
* **Domain-aware query rewriting** (adds astrophysics synonyms when needed)
* **Semantic summarization** (preserves physics terminology)
* **Focused prompt** (prevents drift into unrelated modules)

---

## 5) Metrics You Can Log

* **Retrieval**: `unique_pages`, `redundancy`, `coverage_chars`
* **Quality (manual)**: faithfulness ✅, completeness ⚖️, relevance 🎯

---

## Usage Tips

* If the answer drifts to general MESA overview, constrain retrieval:

  * Set `must_contain=["nuclear","reaction","burning","energy"]` in `real_rag_v2()`.
* If nuclear sections are long, increase reranker window to `[:700]`.
* For heavier models, ensure a GPU runtime in Colab.

---
