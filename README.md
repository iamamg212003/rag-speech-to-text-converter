This project is a Retrieval-Augmented Generation (RAG) system built to answer questions directly from video transcripts using local AI models. Instead of returning large or repeated chunks of text, the system retrieves only the most relevant transcript segments and generates short, precise answers.

The pipeline converts video transcripts into meaningful chunks, transforms them into vector embeddings, and stores them locally. When a user asks a question, the system performs semantic search using cosine similarity to identify the most relevant chunks and then uses a local LLM via Ollama to generate a contextual and accurate answer.

The entire system runs fully offline, making it cost-free, privacy-friendly, and independent of cloud APIs.

Tech Stack:
Python
SentenceTransformers (MiniLM)
NumPy & scikit-learn
Ollama (TinyLlama)
Pickle for local vector storage
