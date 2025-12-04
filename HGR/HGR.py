!pip install -qU langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu pypdf transformers accelerate langchain-text-splitters scikit-learn

import json, torch, numpy as np
from google.colab import files
from sklearn.cluster import AgglomerativeClustering
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# --------------------------
# Upload and Chunk PDF
# --------------------------

uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]
docs = PyPDFLoader(pdf_path).load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900, chunk_overlap=180, add_start_index=True
)
chunks = splitter.split_documents(docs)
for ch in chunks:
    ch.metadata["page"] = ch.metadata.get("page", ch.metadata.get("source", ""))

print(f"Prepared {len(chunks)} chunks")

# --------------------------
# Embed chunks
# --------------------------

emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embs = emb_model.embed_documents([c.page_content for c in chunks])
embs = np.array(embs)

# --------------------------
# Build Hierarchical Tree
# --------------------------

# Cluster into a fixed number of coarse macro-clusters
N_ROOT_CLUSTERS = 6
root_cluster = AgglomerativeClustering(n_clusters=N_ROOT_CLUSTERS).fit_predict(embs)

tree = {}  # root_id -> children clusters -> leaf documents
cluster_to_indices = {}

for idx, c in enumerate(root_cluster):
    cluster_to_indices.setdefault(c, []).append(idx)

# For each macro-cluster, create subclusters
N_SUBCLUSTERS = 3
for root_id, indices in cluster_to_indices.items():
    sub_embs = embs[indices]
    sub = AgglomerativeClustering(n_clusters=min(N_SUBCLUSTERS, len(indices))).fit_predict(sub_embs)
    tree[root_id] = {}
    for sub_id, doc_index in zip(sub, indices):
        tree[root_id].setdefault(sub_id, []).append(doc_index)

print("Tree built ✓ (macro-level + subclusters)")

# --------------------------
# LLM Navigator (Cluster Selector)
# --------------------------

selector = pipeline(
    "text-generation",
    model="google/flan-t5-large",
    device=0 if torch.cuda.is_available() else -1
)

def llm_select_cluster(query, cluster_summaries):
    prompt = f"""
You are a retrieval agent. Select the MOST relevant cluster for answering the query.

Query:
{query}

Cluster summaries:
{json.dumps(cluster_summaries, indent=2)}

Respond ONLY with the number of the most relevant cluster.
"""
    out = selector(prompt, max_length=80, do_sample=False)[0]["generated_text"]
    digits = [int(s) for s in out.split() if s.isdigit()]
    return digits[0] if digits else 0

# --------------------------
# HGR Search
# --------------------------

def summarize_cluster(indices):
    sample = [chunks[i].page_content[:350] for i in indices[:3]]
    return " ".join(sample)

def hgr_retrieve(query, k_final=6):
    # 1. Build summaries of macro clusters
    macro_summaries = {
        i: summarize_cluster(indices) for i, indices in cluster_to_indices.items()
    }
    root_choice = llm_select_cluster(query, macro_summaries)

    # 2. For selected macro cluster, build summaries of subclusters
    sub_summaries = {
        j: summarize_cluster(indices) for j, indices in tree[root_choice].items()
    }
    sub_choice = llm_select_cluster(query, sub_summaries)

    # 3. Retrieve final docs from chosen subcluster
    selected_docs = [chunks[i] for i in tree[root_choice][sub_choice]]

    # pick top k by simple cosine similarity for now
    q_emb = np.array(emb_model.embed_query(query))
    doc_embs = embs[tree[root_choice][sub_choice]]
    scores = np.dot(doc_embs, q_emb) / (np.linalg.norm(doc_embs, axis=1)+1e-6)
    top = np.argsort(scores)[-k_final:][::-1]
    return [selected_docs[i] for i in top]

# --------------------------
# Summarization (same as Advanced RAG)
# --------------------------

def summarize_context(docs, max_chars=2500):
    docs = sorted(docs, key=lambda d: (d.metadata.get("page", 9999), d.metadata.get("start_index", 9999)))
    merged = " ".join([d.page_content for d in docs])
    merged = merged[:max_chars]

    summarizer = pipeline("text2text-generation", model="google/flan-t5-large",
                          device=0 if torch.cuda.is_available() else -1)
    prompt = f"Summarize the astrophysical content, preserving technical detail:\n\n{merged}\n\nSummary:"
    return summarizer(prompt, max_length=420, do_sample=False)[0]["generated_text"]

# --------------------------
# HGR — Run Example Query
# --------------------------

query = "Explain how MESA handles energy generation through nuclear reaction networks, and how the treatment differs for massive vs. low-mass stars."

retrieved = hgr_retrieve(query)
context = summarize_context(retrieved)

print("=== HGR — Context Extracted ===\n")
print(context[:1500])
print("\nPages selected:", [r.metadata.get("page") for r in retrieved])

qa_prompt = f"""
You are an expert astrophysicist. Using ONLY the retrieved context below,
explain how MESA models energy generation through nuclear reaction networks
and how the treatment differs for massive vs low-mass stars.

Context:
{context}

Question:
{query}

Answer:
"""

answer_gen = pipeline("text2text-generation", model="google/flan-t5-xl",
                      device=0 if torch.cuda.is_available() else -1)
final_answer = answer_gen(qa_prompt, max_length=700, do_sample=False)[0]["generated_text"]

print("\n=== FINAL ANSWER (HGR) ===\n")
print(final_answer)
