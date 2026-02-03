import pickle
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "hf_models/all-MiniLM-L6-v2"
OLLAMA_MODEL = "tinyllama"   # you already have this

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading embeddings...")
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

# IMPORTANT: your embeddings.pkl must be a dict like:
# {"chunks": [...], "embeddings": [...]}
chunks = data["chunks"]
embeddings = np.array(data["embeddings"])

print("Ready. Ask questions (type 'exit' to quit).")

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-3:][::-1]

    # 🔹 Combine top chunks into context
    context = "\n".join([chunks[i] for i in top_indices])

    # 🔹 Prompt for LLM
    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question: {query}
Answer:
"""

    # 🔹 Call Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()
    print("\nFINAL ANSWER:\n")
    print(result["response"])
