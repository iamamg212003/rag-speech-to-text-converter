import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = "hf_models/all-MiniLM-L6-v2"
EMBED_FILE = "embeddings.pkl"

TOP_K = 3
MIN_SCORE = 0.3

print("Loading local SentenceTransformer model...")
model = SentenceTransformer(MODEL_PATH)
print("Model loaded!")

print("Loading embeddings...")
with open(EMBED_FILE, "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
embeddings = np.array(data["embeddings"])

print(f"Loaded {len(chunks)} embeddings.")

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])
    sims = cosine_similarity(query_embedding, embeddings)[0]

    ranked = sims.argsort()[::-1]

    print("\nTop results:\n")

    shown = set()
    count = 0

    for idx in ranked:
        if sims[idx] < MIN_SCORE:
            continue

        chunk = chunks[idx]

        if chunk in shown:
            continue

        print(f"Score: {sims[idx]:.3f}")
        print(chunk[:400])  # prevent huge walls of text
        print("-" * 60)

        shown.add(chunk)
        count += 1

        if count == TOP_K:
            break

    if count == 0:
        print("No strong match found.")
