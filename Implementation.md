# 📚 Custom RAG Knowledge Bot - Beginner Friendly Guide

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** bot using your own custom documents. It uses local documents (PDFs, TXT files) and lets you query them in natural language using a simple web interface.

---

## ✅ What This Project Does

1. Loads documents from a folder.
2. Splits them into manageable chunks.
3. Converts them into embedding vectors.
4. Uses FAISS for similarity-based retrieval.
5. Feeds relevant chunks to a local language model to generate answers.
6. Presents it all via a Streamlit app.

---

## 🧰 Libraries Used & Why

### 1. **`os`, `pickle`, `numpy`**

* **`os`**: To read files from folders.
* **`pickle`**: loads the previously saved chunks
* **`numpy`**: Used to handle embedding vectors as arrays for similarity search.

### 2. **`faiss`**

* Facebook AI Similarity Search — a fast library to do nearest neighbor search on high-dimensional vectors (used for document retrieval).

### 3. **`streamlit`**

* To build a simple interactive web app where users can type questions and view answers.

### 4. **`sentence_transformers`**

* Provides pretrained models like `all-MiniLM-L6-v2` for converting text into embedding vectors (numerical representations).

### 5. **`langchain`**

* **`PyMuPDFLoader`**: Load PDF files.
* **`TextLoader`**: Load TXT files.
* **`RecursiveCharacterTextSplitter`**: Splits documents into overlapping chunks of text.

### 6. **`transformers`**

* **`AutoTokenizer`** & **`AutoModelForSeq2SeqLM`**: Load a local pre-trained text generation model like `t5-small` to generate natural language answers.

---

## 🔄 Full Flow with Code Explanation

### 1. **Load Documents**

```python
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        loader = PyMuPDFLoader(path)
    elif filename.endswith(".txt"):
        loader = TextLoader(path)
```

👉 This loads all PDF or TXT documents into memory for processing.

---

### 2. **Split into Chunks**

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)
```

👉 This breaks text into manageable overlapping chunks. Each chunk has 300 characters with 50-character overlap for better context.

---

### 3. **Convert to Embeddings**

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts)
```

👉 Embeds each chunk into a dense vector — this lets us compare how "similar" a chunk is to the user's question.

---

### 4. **Build FAISS Index**

```python
index = faiss.IndexFlatL2(dim)
index.add(np.array(vectors))
```

👉 This stores all embeddings so we can quickly find the most similar chunks to any query.

---

### 5. **Querying (Retrieval + Generation)**

```python
query_vector = embed_model.encode([query])
scores, indices = index.search(query_vector, k)
context = "\n".join([texts[i] for i in indices[0]])
```

👉 Finds the top `k` similar chunks to the user's query.

```python
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
inputs = tokenizer(prompt)
outputs = generator.generate(inputs.input_ids)
```

👉 Feeds the retrieved chunks and question into a T5 model to generate an answer.

---

### 6. **Streamlit UI**

```python
st.title("📚 Custom RAG Knowledge Bot")
query = st.text_input("Ask a question:")
if query:
    answer = rag_pipeline(query)
    st.write(answer)
```

👉 Presents a simple interface for users to type questions and view results.

---

## 📂 Folder Structure

```
project/
│
├── docs/                      # Folder with your PDF and TXT documents
├── local_models/t5-small/     # Local downloaded T5 model files
├── app.py                     # Main Streamlit app file (runnable)
└── README.md                  # This documentation
```

---

## ▶️ How to Run the App

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place documents in the `docs/` folder.
3. Download and place the T5 model (`t5-small`) inside `local_models/t5-small/`.
4. Run the app:

```bash
streamlit run app.py
```

---

## 🔌 Offline Mode

We use:

```python
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

To ensure the HuggingFace model loads from local files.

