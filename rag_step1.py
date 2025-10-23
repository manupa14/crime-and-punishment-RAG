

from pathlib import Path
import json
from typing import List, Dict

import numpy as np
from joblib import dump, load
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import requests


# ======= Config you can tweak =======
PROJECT_DIR = Path(__file__).parent  
BOOK_PATH = PROJECT_DIR / "crime_and_punishment.txt"  
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

CHUNK_SIZE = 800
OVERLAP = 120
TOP_K = 5

# Embedding model (fast, 384-dim)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM (Ollama)
USE_LLM = True
OLLAMA_MODEL = "qwen2:7b"      
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_GEN_URL  = "http://localhost:11434/api/generate"
TEMPERATURE = 0.0
SIMILARITY_REFUSE_THRESHOLD = 0.55

# UX
INTERACTIVE = True  # True = prompt() loop; False = run QUERY_DEFAULT once
QUERY_DEFAULT = "Why does Raskolnikov think he's justified in the murder?"
SHOW_CONTEXT = False  # also print the joined context

# ======= Artifact paths =======
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.jsonl"
EMB_PATH = ARTIFACTS_DIR / "embeddings.npy"
NN_PATH = ARTIFACTS_DIR / "nn.joblib"
META_PATH = ARTIFACTS_DIR / "meta.json"

# ======= Global cache for embedder =======
_EMBEDDER = None


# ======= Core utilities =======
def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Book file not found: {path}")
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return "\n".join(line.strip() for line in txt.splitlines())


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Dict]:
    """
    Greedy paragraph-aware chunker with overlap. Returns dicts: {id, start, end, text}
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = []
    count = 0
    idx = 0

    def flush(buffer_text: str):
        nonlocal idx
        start = 0 if not chunks else chunks[-1]["end"]
        end = start + len(buffer_text)
        chunks.append({"id": idx, "start": start, "end": end, "text": buffer_text})
        idx += 1

    for p in paras:
        if not buf:
            buf.append(p)
            count = len(p)
            continue
        if count + 2 + len(p) <= chunk_size:
            buf.append(p)
            count += 2 + len(p)
        else:
            flush("\n\n".join(buf))
            tail = buf[-1]
            tail_keep = tail[-overlap:] if overlap < len(tail) else tail
            buf = [tail_keep, p]
            count = len(tail_keep) + 2 + len(p)

    if buf:
        flush("\n\n".join(buf))

    # Recompute cumulative spans for readability
    pos = 0
    for c in chunks:
        c_len = len(c["text"])
        c["start"] = pos
        c["end"] = pos + c_len
        pos += c_len
    return chunks


def get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        print(f"Loading embedder: {MODEL_NAME}")
        _EMBEDDER = SentenceTransformer(MODEL_NAME)
    return _EMBEDDER


def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    # normalize_embeddings=True makes cosine behave nicely (dot ~ cosine)
    return np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=True))


def build_index(vectors: np.ndarray) -> NearestNeighbors:
    nn = NearestNeighbors(n_neighbors=min(TOP_K, 10), metric="cosine")
    nn.fit(vectors)
    return nn


def pretty_snippet(s: str, max_chars=320) -> str:
    s = " ".join(s.split())
    return s[: max_chars - 3] + "..." if len(s) > max_chars else s


# ======= Persistence =======
def save_artifacts(chunks, embeddings, nn, book_path: Path):
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(EMB_PATH, embeddings)
    dump(nn, NN_PATH)
    META_PATH.write_text(json.dumps({
        "book_path": str(book_path.resolve()),
        "book_mtime": book_path.stat().st_mtime,
        "model_name": MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "embedding_dim": int(embeddings.shape[1])
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def load_artifacts():
    if not (CHUNKS_PATH.exists() and EMB_PATH.exists() and NN_PATH.exists() and META_PATH.exists()):
        return None
    chunks = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    embeddings = np.load(EMB_PATH)
    nn = load(NN_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return chunks, embeddings, nn, meta


def needs_rebuild(book_path: Path) -> bool:
    art = load_artifacts()
    if art is None:
        return True
    _, _, _, meta = art
    try:
        current_mtime = book_path.stat().st_mtime
    except FileNotFoundError:
        return True
    # Rebuild if book changed or config/model changed
    if abs(current_mtime - meta.get("book_mtime", 0)) > 1e-6:
        return True
    if meta.get("model_name") != MODEL_NAME:
        return True
    if meta.get("chunk_size") != CHUNK_SIZE or meta.get("overlap") != OVERLAP:
        return True
    return False


# ======= Build pipeline =======
def build_pipeline():
    print(f"Reading: {BOOK_PATH}")
    text = read_text(BOOK_PATH)
    print("Chunking...")
    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
    print(f"Chunks: {len(chunks)}")

    model = get_embedder()
    print("Embedding chunks...")
    embeddings = embed_texts([c["text"] for c in chunks], model)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Fitting NearestNeighbors (cosine)...")
    nn = build_index(embeddings)

    print("Saving artifacts...")
    save_artifacts(chunks, embeddings, nn, BOOK_PATH)
    print("✅ Build complete.")


# ======= LLM (Ollama) =======
def build_messages(question: str, docs: list[str]) -> list[dict]:
    """
    Build a strict prompt. Retrieved text is placed in user content and clearly delimited.
    """
    context = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs, 1))
    system = (
        "You are a careful assistant. Use ONLY the provided context to answer.\n"
        "If the context is insufficient or missing, say you don't know.\n"
        "Cite sources inline like [1], [2] referring to the context chunks.\n"
        "Do not fabricate or use outside knowledge."
    )
    user = f"Question:\n{question}\n\nContext:\n---BEGIN CONTEXT---\n{context}\n---END CONTEXT---"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def _compose_prompt_for_generate(messages: list[dict]) -> str:
    # Convert chat-style messages to a single prompt string
    sys_parts = [m["content"] for m in messages if m["role"] == "system"]
    usr_parts = [m["content"] for m in messages if m["role"] == "user"]
    sys_text = "\n\n".join(sys_parts).strip()
    user_text = "\n\n".join(usr_parts).strip()
    return f"<<SYS>>\n{sys_text}\n<</SYS>>\n\n{user_text}"


def answer_with_ollama(messages: list[dict]) -> str:
    # Try /api/chat first (newer Ollama). If 404, fallback to /api/generate.
    try:
        r = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": TEMPERATURE},
            },
            timeout=120,
        )
        if r.status_code == 404:
            raise NotImplementedError("chat endpoint not available")
        r.raise_for_status()
        data = r.json()
        # Standard Ollama chat shape
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"].strip()
        # Some frontends mimic OpenAI
        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        return "(No content from /api/chat.)"
    except Exception:
        # Fallback: /api/generate expects a single prompt string
        prompt = _compose_prompt_for_generate(messages)
        rg = requests.post(
            OLLAMA_GEN_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": TEMPERATURE},
            },
            timeout=120,
        )
        rg.raise_for_status()
        j = rg.json()
        if "response" in j:
            return j["response"].strip()
        if "message" in j and "content" in j["message"]:
            return j["message"]["content"].strip()
        return "(No content from /api/generate.)"


# ======= Query pipeline =======
def run_query(q: str):
    art = load_artifacts()
    if art is None:
        raise RuntimeError("Artifacts missing. Run build_pipeline() first.")
    chunks, embeddings, nn, meta = art

    model = get_embedder()
    q_vec = embed_texts([q], model)
    distances, indices = nn.kneighbors(q_vec, n_neighbors=TOP_K, return_distance=True)

    print("\n=== Top matches ===")
    best_sim = 0.0
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        sim = 1.0 - float(dist)  # cosine similarity
        best_sim = max(best_sim, sim)
        ch = chunks[int(idx)]
        print(f"[{rank}] sim={sim:.4f} id={ch['id']} span=({ch['start']}-{ch['end']})")
        print(pretty_snippet(ch["text"]))
        print("")

    docs_for_prompt = [chunks[int(i)]["text"] for i in indices[0]]

    if SHOW_CONTEXT:
        context = "\n\n".join(docs_for_prompt)
        print("=== Context (joined) ===\n")
        print(context)

    if USE_LLM:
        if best_sim < SIMILARITY_REFUSE_THRESHOLD:
            print("=== LLM Answer ===\n")
            print("I don't know. The retrieved context is not relevant enough to answer confidently.")
            return
        msgs = build_messages(q, docs_for_prompt)
        print("=== LLM Answer ===\n")
        print(answer_with_ollama(msgs))


# ======= Main (no CLI) =======
if __name__ == "__main__":
    # Auto-build if needed (first run or book/config changed)
    if needs_rebuild(BOOK_PATH):
        print("Artifacts are missing or stale → rebuilding...")
        build_pipeline()
    else:
        print("Artifacts are fresh. Skipping rebuild.")

    if INTERACTIVE:
        print("\nType your query (empty line to exit):")
        while True:
            q = input("> ").strip()
            if not q:
                break
            run_query(q)
    else:
        print(f"\nRunning default query: {QUERY_DEFAULT!r}")
        run_query(QUERY_DEFAULT)
