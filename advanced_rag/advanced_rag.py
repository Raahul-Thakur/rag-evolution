!pip install -qU langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu pypdf rank_bm25 transformers accelerate langchain-text-splitters
import re, math, numpy as np
from google.colab import files
from collections import defaultdict
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]

docs = PyPDFLoader(pdf_path).load()
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=180, add_start_index=True)
chunks = splitter.split_documents(docs)

for ch in chunks:
    ch.metadata["page"] = ch.metadata.get("page", ch.metadata.get("source", ""))

print(f"Prepared {len(chunks)} chunks")

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vs = FAISS.from_documents(chunks, embedding=emb)
bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 12

rerank_model_id = "cross-encoder/ms-marco-MiniLM-L-12-v2"
tok = AutoTokenizer.from_pretrained(rerank_model_id)
rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_id)
rerank_model.eval()
if torch.cuda.is_available():
    rerank_model.cuda()

def cross_encoder_score(query, passage):
    inputs = tok([query], [passage], padding=True, truncation=True, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        scores = rerank_model(**inputs).logits.squeeze(-1)
    return float(scores.item())

def rewrite(q: str):
    q2 = q.strip().rstrip("?")
    expansions = {
        "mesa": ["Modules for Experiments in Stellar Astrophysics"],
        "stellar": ["star", "stars", "stellar structure", "stellar evolution"],
        "module": ["component", "library", "subroutine", "package"],
        "evolution": ["evolutionary track", "time evolution", "evolutionary model"],
        "diffusion": ["mixing", "chemical diffusion", "composition transport"],
        "rotation": ["angular momentum", "spin", "rotational velocity", "rotational mixing"],
        "convection": ["convective zone", "energy transport"],
        "accretion": ["mass transfer", "accreting", "infall", "mass gain"],
        "white dwarf": ["WD", "degenerate star", "compact remnant"],
        "numerical": ["computational", "solver", "algorithmic"],
        "test suite": ["validation suite", "verification test", "code comparison"],
        "physics": ["microphysics", "macrophysics", "equation of state", "opacity"],
    }

    extra = []
    for k, vals in expansions.items():
        if re.search(rf"\b{k}\b", q2, flags=re.I):
            extra += vals
    if extra:
        q2 = q2 + " " + " ".join(set(extra))
    return q2

def metadata_filter(docs, *, page_range=None, must_contain=None):
    out = []
    for d in docs:
        ok = True
        if page_range is not None and isinstance(d.metadata.get("page"), int):
            ok = ok and (page_range[0] <= d.metadata["page"] <= page_range[1])
        if must_contain:
            ok = ok and any(s.lower() in d.page_content.lower() for s in must_contain)
        if ok:
            out.append(d)
    return out

def summarize_context(text, model="google/flan-t5-large"):
    """LLM-based summarization preserving technical details."""
    summarizer = pipeline("text2text-generation", model=model, device=0 if torch.cuda.is_available() else -1)
    prompt = f"Summarize the following astrophysical text while keeping equations, processes, and key terminology:\n\n{text}\n\nSummary:"
    return summarizer(prompt, max_length=350, do_sample=False)[0]["generated_text"]

def order_and_summarize(docs, max_chars=2500):
    """Order chunks and summarize them while maintaining physical coherence."""
    docs = sorted(docs, key=lambda d: (d.metadata.get("page", 9999), d.metadata.get("start_index", 9999)))

    merged = []
    current, cur_page = "", None
    for d in docs:
        pg = d.metadata.get("page")
        if pg == cur_page:
            current += " " + d.page_content
        else:
            if current:
                merged.append(current)
            current, cur_page = d.page_content, pg
    if current:
        merged.append(current)

    full_text = " ".join(merged)
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]

    return summarize_context(full_text)

def hybrid_retrieve(query, alpha=0.7, k_dense=10):
    """Combine BM25 and FAISS with weighting for technical recall."""
    sparse_docs = bm25.invoke(query)
    dense_docs = vs.similarity_search(query, k=k_dense)

    scores = defaultdict(float)
    for d in sparse_docs:
        scores[id(d)] += (1 - alpha)
    for d in dense_docs:
        scores[id(d)] += alpha

    merged = []
    seen = set()
    for d in sparse_docs + dense_docs:
        key = (d.metadata.get("page"), d.metadata.get("start_index"))
        if key not in seen:
            seen.add(key)
            merged.append(d)
    return merged

def eval_bundle(docs):
    pages = [d.metadata.get("page") for d in docs]
    unique_pages = len(set(pages))
    coverage_chars = sum(len(d.page_content) for d in docs)
    redundancy = 1 - unique_pages / max(1, len(docs))
    return {
        "k": len(docs),
        "unique_pages": unique_pages,
        "redundancy": round(redundancy, 3),
        "coverage_chars": coverage_chars
    }

def real_rag(query, *, page_range=None, must_contain=None, k_dense=10, k_final=6):
    q_rew = rewrite(query)
    merged = hybrid_retrieve(q_rew, alpha=0.7, k_dense=k_dense)
    merged = metadata_filter(merged, page_range=page_range, must_contain=must_contain)

    scored = [(d, cross_encoder_score(q_rew, d.page_content[:512])) for d in merged]
    scored.sort(key=lambda x: x[1], reverse=True)
    topk = [d for d, s in scored[:k_final]]

    context = order_and_summarize(topk, max_chars=2500)
    stats = eval_bundle(topk)
    return context, stats, topk

query = "Explain how MESA handles energy generation through nuclear reaction networks, and how the treatment differs for massive vs. low-mass stars."
context, stats, retrieved = real_rag(query)

print("=== REAL RAG — Ordered & Summarized Context ===\n")
print(context[:1500])
print("\n--- Retrieval Stats ---")
print(stats)
print("\nPages picked:", [r.metadata.get("page") for r in retrieved])

qa_prompt = f"""
You are an expert astrophysicist. Using only the context from the MESA documentation below,
explain how MESA models energy generation through nuclear reaction networks,
and highlight differences in treatment between massive and low-mass stars.
Be factual, concise, and avoid speculation.

Context:
{context}

Question:
{query}

Answer:
"""

llm = pipeline("text2text-generation", model="google/flan-t5-xl", device=0 if torch.cuda.is_available() else -1)
answer = llm(qa_prompt, max_length=700, do_sample=False, temperature=0.3)[0]["generated_text"]

print("\n=== FINAL ANSWER (REAL RAG v2) ===\n")
print(answer)

