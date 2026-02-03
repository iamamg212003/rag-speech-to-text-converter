import os
import pickle
from sentence_transformers import SentenceTransformer

MODEL_PATH = "hf_models/all-MiniLM-L6-v2"
CHUNK_DIR = "chunks"
OUTPUT_FILE = "embeddings.pkl"

print("Loading local SentenceTransformer model...")
model = SentenceTransformer(MODEL_PATH)
print("Model loaded!")

chunks = []

for fname in sorted(os.listdir(CHUNK_DIR)):
    if fname.endswith(".txt"):
        with open(os.path.join(CHUNK_DIR, fname), "r", encoding="utf-8") as f:
            # split into individual chunks
            file_chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]
            chunks.extend(file_chunks)

print(f"Found {len(chunks)} real chunks.")

print("Embedding chunks locally...")
embeddings = model.encode(chunks, batch_size=8, show_progress_bar=True)

data = {
    "chunks": chunks,
    "embeddings": embeddings
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)

print("Embeddings saved correctly!")
