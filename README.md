🛠️ Project Workflow (Step-by-Step)

This project converts video content into searchable knowledge using Speech-to-Text and Retrieval-Augmented Generation (RAG). The workflow is divided into clear stages:

🔹 Step 1: Video to Audio
Input videos are processed to extract audio.
This makes speech-to-text transcription easier and faster.

Script used:
process_videos.py

🔹 Step 2: Speech to Text (Transcription)
Audio files are converted into text using a speech-to-text model.
Each video generates a corresponding transcript file.

Script used:
stt.py

Output:

transcripts/
 ├── video1.txt
 ├── video2.txt

🔹 Step 3: Text Chunking
Large transcripts are split into smaller, meaningful chunks.
Chunking improves semantic search and prevents huge responses.

Script used:
chunker.py

Output:

chunks/
 ├── video1_chunks.txt
 ├── video2_chunks.txt

🔹 Step 4: Embedding Generation
Each text chunk is converted into a vector embedding using a SentenceTransformer model.
These embeddings represent the semantic meaning of chunks.

Script used:
embedder.py

Output:
embeddings.pkl

🔹 Step 5: Semantic Retrieval
When a user asks a question:
The query is converted into an embedding.
Cosine similarity is used to find the most relevant chunks.

Script used:
retriever.py

🔹 Step 6: RAG-based Answer Generation
Retrieved chunks are passed as context to an LLM (Ollama local model).
The LLM generates a concise and accurate answer.

Script used:
rag_llm.py
rag_answer.py

🧠 Key Concepts Used
Speech-to-Text (STT)
Text Chunking
Sentence Embeddings
Cosine Similarity Search
Retrieval-Augmented Generation (RAG)
Local LLM using Ollama

⚙️ Tech Stack
Python
OpenAI Whisper / STT model
SentenceTransformers (all-MiniLM-L6-v2)
Scikit-learn
Ollama (local LLM: TinyLlama / Phi / Mistral)
NumPy

📌 Features
✔ Converts video to searchable text
✔ Answers questions from video content
✔ No cloud API required
✔ Uses local LLM
✔ Precise chunk-based retrieval
✔ Lightweight embeddings

🚀 How to Run
python process_videos.py
python stt.py
python chunker.py
python embedder.py
python rag_answer.py

🧾 Notes
Large files (models, videos, embeddings) are ignored using .gitignore.
Only source code is tracked in GitHub.
Ollama must be running in background.
