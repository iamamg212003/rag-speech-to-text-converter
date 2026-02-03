import whisper
import os

VIDEO_DIR = "videos"
OUTPUT_DIR = "transcripts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = whisper.load_model("base", download_root="./models")

for file in os.listdir(VIDEO_DIR):
    if file.lower().endswith((".mp4", ".webm", ".mkv")):
        file_path = os.path.join(VIDEO_DIR, file)
        print(f"Transcribing: {file}")

        result = model.transcribe(file_path)

        output_file = os.path.splitext(file)[0].strip() + ".txt"
        output_path = os.path.join(OUTPUT_DIR, output_file)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"Saved: {output_path}")

print("All videos transcribed!")
