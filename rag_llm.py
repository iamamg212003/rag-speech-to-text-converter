import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "tinyllama"   # or mistral if RAM allows

def ask_llm(question, context):
    prompt = f"""
You are a helpful assistant. Answer ONLY from the context.

Context:
{context}

Question:
{question}
"""

    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    res = requests.post(OLLAMA_URL, json=data)
    return res.json()["response"]
