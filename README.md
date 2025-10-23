# Minimal Local RAG (Python + Sentence-Transformers + scikit-learn + Ollama)

A tiny Retrieval-Augmented Generation pipeline that runs **entirely locally**:
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Index:** `sklearn.neighbors.NearestNeighbors` with cosine distance
- **LLM:** **Ollama** (e.g., `qwen2:7b-instruct`, `qwen2.5:7b-instruct`, or `llama3.2:3b`)

> Designed to be understandable end-to-end. No frameworks, no CLIs to learn.

---

## How it works (in one screen)

1. Load `crime_and_punishment.txt`  
2. Chunk into overlapping segments (default: 800 chars, 120 overlap)  
3. Embed chunks → vectors (normalized)  
4. Fit a cosine **NearestNeighbors** index  
5. At query time: embed the query → get top-k chunks → **paste** those raw chunks into the LLM prompt → generate answer  
6. Refuse if retrieval looks weak (similarity threshold)



---

## Repo layout

## Requirements

- Python 3.9+  
- Windows/macOS/Linux
- [Ollama](https://ollama.com) running locally (model selected in variable definition)

Python deps:
```bash
pip install numpy scikit-learn sentence-transformers joblib requests
