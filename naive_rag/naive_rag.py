import warnings
warnings.filterwarnings('ignore')

!pip install -qU langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu pypdf langchain-text-splitters

from google.colab import files
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]

docs = PyPDFLoader(pdf_path).load()
print(f"Loaded {len(docs)} pages from {pdf_path}")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vs = FAISS.from_documents(chunks, embedding=emb)
retriever = vs.as_retriever(search_kwargs={"k": 5})

query = "What physical processes does MESA include to model stellar interiors"
hits = retriever.invoke(query)

print("\n=== Top Results (Naive RAG) ===")
context_texts = []
for i, h in enumerate(hits, 1):
    print(f"\n[{i}] p.{h.metadata.get('page', 'NA')} — {h.page_content[:600]}...")
    context_texts.append(h.page_content)

context = "\n\n".join(context_texts)

qa_prompt = f"Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
llm = pipeline("text2text-generation", model="google/flan-t5-base")
answer = llm(qa_prompt, max_length=512, do_sample=False)[0]["generated_text"]

print("\n=== Final Answer ===")
print(answer)

