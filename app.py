import os
import pickle
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- CONFIGURATION ---
FOLDER_PATH = r"C:\Users\my pc\PycharmProjects\pythonProject11\docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL_PATH = "./local_models/t5-small"
INDEX_FILE = "vector_index.faiss"
TEXTS_FILE = "texts.pkl"

# --- STEP 1: LOAD DOCUMENTS ---
def load_documents(folder_path):
    all_documents = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(path)
        elif filename.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        docs = loader.load()
        all_documents.extend(docs)
    return all_documents

# --- STEP 2: SPLIT DOCUMENTS ---
def split_documents(documents, chunk_size=100, chunk_overlap=10):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks]

# --- STEP 3: EMBED TEXTS ---
def embed_texts(texts, model_name):
    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, show_progress_bar=True)
    return model, vectors

# --- STEP 4: BUILD/LOAD INDEX ---
def build_faiss_index(vectors, dim):
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))
    return index

def save_index(index, file_path):
    faiss.write_index(index, file_path)

def load_index(file_path):
    return faiss.read_index(file_path)

# --- STEP 5: RETRIEVE PASSAGES ---
def retrieve_passages(query, k, embed_model, texts, index):
    query_vector = embed_model.encode([query])
    scores, indices = index.search(np.array(query_vector), k)
    return [texts[i] for i in indices[0]]

# --- STEP 6: GENERATE ANSWER ---
def generate_answer(query, context, tokenizer, generator):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = generator.generate(inputs.input_ids, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- RAG PIPELINE ---
def rag_pipeline(query, k=5):
    top_chunks = retrieve_passages(query, k, embedding_model, texts, index)
    context = "\n".join(top_chunks)
    return generate_answer(query, context, tokenizer, generator)

# --- MAIN EXECUTION ---
if not os.path.exists(INDEX_FILE) or not os.path.exists(TEXTS_FILE):
    print("Processing documents...")
    docs = load_documents(FOLDER_PATH)
    texts = split_documents(docs)
    embedding_model, vectors = embed_texts(texts, EMBEDDING_MODEL_NAME)
    index = build_faiss_index(vectors, vectors.shape[1])
    save_index(index, INDEX_FILE)
    with open(TEXTS_FILE, "wb") as f:
        pickle.dump(texts, f)
else:
    print("Loading precomputed index and texts...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    index = load_index(INDEX_FILE)
    with open(TEXTS_FILE, "rb") as f:
        texts = pickle.load(f)

# Force offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Load generator from local path
tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_PATH, local_files_only=True)
generator = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_PATH, local_files_only=True)


# --- UI (Streamlit) ---
st.title("📚 Custom RAG Knowledge Bot")
query = st.text_input("Ask a question:")
if query:
    answer = rag_pipeline(query)
    st.markdown("**Answer:**")
    st.write(answer)
