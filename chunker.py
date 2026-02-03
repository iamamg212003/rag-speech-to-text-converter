import os
from pathlib import Path

TRANSCRIPTS_DIR = "transcripts"
CHUNKS_DIR = "chunks"

CHUNK_SIZE = 200
OVERLAP = 30


def chunk_text(text, chunk_size=200, overlap=30):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def main():
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    for file in os.listdir(TRANSCRIPTS_DIR):
        if file.endswith(".txt"):
            transcript_path = os.path.join(TRANSCRIPTS_DIR, file)

            with open(transcript_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            chunks = chunk_text(text)

            base_name = Path(file).stem
            output_file = os.path.join(CHUNKS_DIR, f"{base_name}_chunks.txt")

            with open(output_file, "w", encoding="utf-8") as out:
                for i, chunk in enumerate(chunks, 1):
                    out.write(chunk + "\n\n")

            print(f"Chunked: {file} → {output_file}")


if __name__ == "__main__":
    main()
