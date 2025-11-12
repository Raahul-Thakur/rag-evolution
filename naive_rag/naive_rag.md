
# Naive RAG Architecture (Colab Implementation)

This document explains the architecture and flow of the **Naive Retrieval-Augmented Generation (RAG)** system implemented in the provided Colab script. Each block below represents a key stage in the pipeline, written in Markdown format for clarity.

---

## 1. Environment Setup

```bash
!pip install -qU langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu pypdf langchain-text-splitters transformers
```

We install the required dependencies:

* **LangChain** → handles document loading, chunking, and retrieval.
* **SentenceTransformers** → used for embedding text.
* **FAISS** → for vector similarity search.
* **Transformers** → provides the text generation model (Flan-T5).

---

## 2. PDF Upload and Loading

```python
from google.colab import files
from langchain_community.document_loaders import PyPDFLoader

uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]
docs = PyPDFLoader(pdf_path).load()
```

* The user uploads a PDF file from the local system.
* `PyPDFLoader` loads all pages as individual **documents**.
* Each document contains text and metadata like page number.

---

## 3. Text Chunking

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
chunks = splitter.split_documents(docs)
```

* Long documents are split into overlapping **chunks** (≈1000 characters each).
* The overlap (200) helps retain continuity between chunks.
* Each chunk is treated as a retrievable unit.

---

## 4. Embedding and Vector Indexing

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vs = FAISS.from_documents(chunks, embedding=emb)
retriever = vs.as_retriever(search_kwargs={"k": 5})
```

* Each chunk is converted into a **vector embedding** using a transformer model.
* These embeddings are stored in a **FAISS vector index**.
* `retriever` retrieves the **top 5** most relevant chunks for a given query.

---

## 5. Query and Retrieval

```python
query = "What physical processes does MESA include to model stellar interiors?"
hits = retriever.invoke(query)
```

* A query is given as natural language text.
* The retriever finds chunks with the most similar meaning (via cosine similarity).
* Results include text content and metadata (like page numbers).

---

## 6. Displaying Retrieved Chunks

```python
for i, h in enumerate(hits, 1):
    print(f"\n[{i}] p.{h.metadata.get('page', 'NA')} — {h.page_content[:600]}...")
```

* Prints the top 5 chunks.
* Shows the first 600 characters of each retrieved text block.

---

## 7. Context Assembly

```python
context_texts = [h.page_content for h in hits]
context = "\n\n".join(context_texts)
```

* Merges all retrieved chunks into a single **context passage**.
* This forms the knowledge base from which the model will generate an answer.

---

## 8. Answer Generation (LLM Stage)

```python
from transformers import pipeline

qa_prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

llm = pipeline("text2text-generation", model="google/flan-t5-base")
answer = llm(qa_prompt, max_length=512, do_sample=False)[0]["generated_text"]
print(answer)
```

* The context and question are combined into a **prompt**.
* A local model (Flan-T5) is used for text generation.
* The LLM outputs a concise **answer** synthesized from the context.

---

## 9. Architectural Flow Summary

```bash
PDF → Loader → Chunker → Embeddings → FAISS Index → Retriever → Context → LLM Answer
```

### Step-by-Step Summary:

1. **Input:** User uploads a PDF.
2. **Preprocessing:** PDF is split into chunks.
3. **Embedding:** Chunks are converted into dense vectors.
4. **Indexing:** FAISS stores these vectors.
5. **Retrieval:** Query retrieves top relevant chunks.
6. **Reasoning:** LLM (Flan-T5) uses those chunks to generate the final answer.

---

**End Result:**
A complete **Naive RAG** pipeline that retrieves relevant knowledge from PDFs and uses a local LLM to synthesize coherent answ
