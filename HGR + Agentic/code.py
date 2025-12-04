!pip install -qU langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu pypdf transformers accelerate langchain-text-splitters scikit-learn rank_bm25

import os, json, time, re, math, numpy as np
from collections import defaultdict
from typing import List, Dict

from google.colab import files
import torch

from sklearn.cluster import AgglomerativeClustering

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =========================
# 1. Upload & Chunk PDF
# =========================

uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]

docs = PyPDFLoader(pdf_path).load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=180,
    add_start_index=True,
    strip_whitespace=True,
)
chunks = splitter.split_documents(docs)

for ch in chunks:
    ch.metadata["page"] = ch.metadata.get("page", ch.metadata.get("source", ""))

print(f"Prepared {len(chunks)} chunks")

# =========================
# 2. Embeddings & Flat Stores
# =========================

emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
emb_array = np.array(emb_model.embed_documents([c.page_content for c in chunks]))

vs = FAISS.from_documents(chunks, embedding=emb_model)
bm25_global = BM25Retriever.from_documents(chunks)
bm25_global.k = 12

# Cross-encoder for reranking
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

# =========================
# 3. Query Rewrite & Intent
# =========================

def rewrite(q: str) -> str:
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

def route_intent(q: str) -> str:
    ql = q.lower()
    if any(w in ql for w in ["define", "what is", "meaning", "concept"]):
        return "definition"
    if any(w in ql for w in ["compare", "versus", "difference"]):
        return "comparison"
    if any(w in ql for w in ["how to", "steps", "pipeline", "implement"]):
        return "howto"
    if any(w in ql for w in ["summarize", "summary", "high level"]):
        return "summary"
    return "generic"

# =========================
# 4. Episodic Memory
# =========================

MEMO_PATH = "hgr_agentic_memory.json"
if not os.path.exists(MEMO_PATH):
    with open(MEMO_PATH, "w") as f:
        json.dump({"episodes": []}, f)

def remember(query: str, answer: str, pages: List[int]):
    with open(MEMO_PATH, "r") as f:
        mem = json.load(f)
    mem["episodes"].append({
        "ts": time.time(),
        "query": query,
        "answer": answer,
        "pages": pages,
    })
    with open(MEMO_PATH, "w") as f:
        json.dump(mem, f)

def episodic_recall(query: str, top_n: int = 2):
    with open(MEMO_PATH, "r") as f:
        mem = json.load(f)
    q_tokens = set(re.findall(r"\w+", query.lower()))
    scored = []
    for ep in mem.get("episodes", [])[-40:]:
        ep_tokens = set(re.findall(r"\w+", ep["query"].lower()))
        jacc = len(q_tokens & ep_tokens) / max(1, len(q_tokens | ep_tokens))
        scored.append((jacc, ep))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ep for score, ep in scored[:top_n] if score > 0]

# =========================
# 5. Build Hierarchical Tree
# =========================

N_ROOT_CLUSTERS = min(6, len(chunks))
root_labels = AgglomerativeClustering(
    n_clusters=N_ROOT_CLUSTERS
).fit_predict(emb_array)

cluster_to_indices: Dict[int, List[int]] = {}
for idx, c in enumerate(root_labels):
    cluster_to_indices.setdefault(c, []).append(idx)

tree: Dict[int, Dict[int, List[int]]] = {}
N_SUBCLUSTERS = 3

for root_id, indices in cluster_to_indices.items():
    if len(indices) <= 2:
        # too small to subcluster
        tree[root_id] = {0: indices}
        continue
    sub_embs = emb_array[indices]
    n_sub = min(N_SUBCLUSTERS, len(indices))
    sub_labels = AgglomerativeClustering(
        n_clusters=n_sub
    ).fit_predict(sub_embs)
    tree[root_id] = {}
    for sub_label, doc_index in zip(sub_labels, indices):
        tree[root_id].setdefault(sub_label, []).append(doc_index)

print("Hierarchical tree built ✓ (macro clusters + subclusters)")

# =========================
# 6. LLM Navigator (Cluster Selection)
# =========================

cluster_selector = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=0 if torch.cuda.is_available() else -1
)

def summarize_indices(indices: List[int], max_snips: int = 3, max_chars: int = 350) -> str:
    samples = []
    for i in indices[:max_snips]:
        txt = chunks[i].page_content.replace("\n", " ")
        samples.append(txt[:max_chars])
    return " ".join(samples)

def llm_select_clusters(query: str, summaries: Dict[int, str], top_m: int = 2) -> List[int]:
    """
    Ask the LLM which clusters are most relevant.
    We allow it to pick up to `top_m` clusters, but we parse digits and truncate.
    """
    prompt = f"""
You are a retrieval planning agent.
Given the user query and short summaries of document clusters,
select up to {top_m} cluster IDs that are most relevant.

Query:
{query}

Clusters (format: id: summary snippet):
{json.dumps(summaries, indent=2)}

Respond ONLY with the IDs as space-separated integers, like:
0
or
1 3
or
2 4
Do not add any explanation.
"""
    out = cluster_selector(prompt, max_length=120, do_sample=False)[0]["generated_text"]
    ids = [int(tok) for tok in re.findall(r"\d+", out)]
    ids = [i for i in ids if i in summaries]
    if not ids:
        # fallback: pick the first cluster
        ids = [sorted(summaries.keys())[0]]
    return ids[:top_m]

# =========================
# 7. Agentic HGR Retrieval
# =========================

def self_reflect(query: str, context: str) -> Dict:
    q_tokens = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
    ctx_tokens = set(re.findall(r"[a-zA-Z]{3,}", context.lower()))
    overlap = len(q_tokens & ctx_tokens) / max(1, len(q_tokens))
    decision = "good" if overlap >= 0.5 else "retry"
    return {"overlap": round(overlap, 3), "decision": decision}

def hybrid_retrieve_local(query: str, candidate_indices: List[int], k_dense: int = 10):
    """
    Hybrid dense + BM25 retrieval restricted to the candidate indices.
    """
    candidate_docs = [chunks[i] for i in candidate_indices]

    # Local BM25 on candidate docs
    bm25_local = BM25Retriever.from_documents(candidate_docs)
    bm25_local.k = min(10, len(candidate_docs))
    sparse_docs = bm25_local.invoke(query)

    # Local dense retrieval (manual similarity over precomputed embeddings)
    q_emb = np.array(emb_model.embed_query(query))
    cand_embs = emb_array[candidate_indices]
    sims = cand_embs @ q_emb / (np.linalg.norm(cand_embs, axis=1) + 1e-8)
    top_idx = np.argsort(sims)[-k_dense:][::-1]
    dense_docs = [candidate_docs[i] for i in top_idx]

    merged = []
    seen_keys = set()
    for d in sparse_docs + dense_docs:
        key = (d.metadata.get("page"), d.metadata.get("start_index"))
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(d)
    return merged

def order_and_summarize(docs, max_chars=2500, model_name="google/flan-t5-large"):
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
    full_text = full_text[:max_chars]

    summarizer = pipeline(
        "text2text-generation",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )

    prompt = f"Summarize the following astrophysical technical text, preserving equations, physical processes, and key terminology:\n\n{full_text}\n\nSummary:"
    return summarizer(prompt, max_length=420, do_sample=False)[0]["generated_text"]

def eval_bundle(docs):
    pages = [d.metadata.get("page") for d in docs]
    unique_pages = len(set(pages))
    coverage_chars = sum(len(d.page_content) for d in docs)
    redundancy = 1 - unique_pages / max(1, len(docs))
    return {
        "k": len(docs),
        "unique_pages": unique_pages,
        "redundancy": round(redundancy, 3),
        "coverage_chars": coverage_chars,
    }

def hgr_agentic_rag(
    query: str,
    k_dense: int = 10,
    k_final: int = 6,
):
    # ---- Agentic Step 1: Rewrite + Intent ----
    q_rew = rewrite(query)
    intent = route_intent(query)

    # ---- Agentic Step 2: LLM-Guided Hierarchical Cluster Navigation ----
    # Macro level
    macro_summaries = {
        rid: summarize_indices(indices)
        for rid, indices in cluster_to_indices.items()
    }
    chosen_macros = llm_select_clusters(q_rew, macro_summaries, top_m=2)

    # Subcluster level
    chosen_candidates = []
    for rid in chosen_macros:
        subclusters = tree[rid]
        sub_summaries = {
            sid: summarize_indices(indices)
            for sid, indices in subclusters.items()
        }
        chosen_subs = llm_select_clusters(q_rew, sub_summaries, top_m=2)
        for sid in chosen_subs:
            chosen_candidates.extend(subclusters[sid])

    # Safety fallback if something goes wrong
    if not chosen_candidates:
        chosen_candidates = list(range(len(chunks)))

    # ---- Agentic Step 3: Hybrid retrieval restricted to chosen clusters ----
    merged = hybrid_retrieve_local(q_rew, chosen_candidates, k_dense=k_dense)

    # ---- Agentic Step 4: Cross-encoder reranking (global relevance) ----
    scored = [(d, cross_encoder_score(q_rew, d.page_content[:512])) for d in merged]
    scored.sort(key=lambda x: x[1], reverse=True)
    topk_docs = [d for d, s in scored[:k_final]]

    # ---- Agentic Step 5: Compose context & reflect ----
    context = order_and_summarize(topk_docs, max_chars=2500)
    reflex = self_reflect(query, context)

    if reflex["decision"] == "retry":
        # If weak overlap, widen search to all subclusters of chosen macros
        wider_indices = []
        for rid in chosen_macros:
            for sid, indices in tree[rid].items():
                wider_indices.extend(indices)
        wider_indices = list(set(wider_indices))
        merged2 = hybrid_retrieve_local(q_rew, wider_indices, k_dense=k_dense + 4)
        scored2 = [(d, cross_encoder_score(q_rew, d.page_content[:512])) for d in merged2]
        scored2.sort(key=lambda x: x[1], reverse=True)
        topk_docs = [d for d, s in scored2[:k_final]]
        context = order_and_summarize(topk_docs, max_chars=2500)
        reflex = self_reflect(query, context)

    stats = eval_bundle(topk_docs)
    pages = [d.metadata.get("page") for d in topk_docs]

    # ---- Agentic Step 6: Final QA with context ----
    qa_prompt = f"""
You are an expert astrophysicist and AI research assistant.
User intent type: {intent}.

Using ONLY the technical context from the MESA documentation below,
answer the user's question. Be precise, factual, and avoid speculation.

Context:
{context}

Question:
{query}

Answer:
"""
    qa_llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-xl",
        device=0 if torch.cuda.is_available() else -1,
    )
    answer = qa_llm(qa_prompt, max_length=800, do_sample=False, temperature=0.3)[0]["generated_text"]

    remember(query, answer, pages)
    return {
        "intent": intent,
        "context": context,
        "answer": answer,
        "stats": stats,
        "pages": pages,
        "reflection": reflex,
    }

# =========================
# 8. Run HGR + Agentic Example
# =========================

query = (
    "Explain how MESA handles energy generation through nuclear reaction "
    "networks, and how the treatment differs for massive vs. low-mass stars."
)

result = hgr_agentic_rag(query)

print("=== HGR + AGENTIC RAG ===\n")
print("Intent:", result["intent"])
print("Reflection:", result["reflection"])
print("\n--- Retrieval Stats ---")
print(result["stats"])
print("Pages picked:", result["pages"])

print("\n=== CONTEXT (COMPOSED) ===\n")
print(result["context"][:1500])

print("\n=== FINAL ANSWER ===\n")
print(result["answer"])

print("\n=== Episodic Memory Peek ===")
print(episodic_recall(query, top_n=1))
