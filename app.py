import os
import pickle
import tempfile

import numpy as np
import streamlit as st
import whisper
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Speech-to-Text Converter",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ RAG Speech-to-Text Converter")
st.write("AI-powered speech-to-text and question answering using RAG.")


# --------------------------------------------------
# Load Whisper
# --------------------------------------------------

@st.cache_resource
def load_whisper():

    return whisper.load_model(
        "tiny",
        download_root="./models"
    )


# --------------------------------------------------
# Load Sentence Transformer
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# Create Embeddings
# --------------------------------------------------

@st.cache_data
def create_embeddings(chunks):

    model = load_embedding_model()

    embeddings = model.encode(
        chunks,
        batch_size=8,
        show_progress_bar=False
    )

    return np.array(embeddings)


# --------------------------------------------------
# Chunk Transcript
# --------------------------------------------------

def chunk_text(
    text,
    chunk_size=200,
    overlap=30
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# File Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an audio or video file",
    type=[
        "mp3",
        "wav",
        "mp4",
        "webm",
        "mkv",
        "m4a"
    ]
)


# --------------------------------------------------
# Transcription
# --------------------------------------------------

if uploaded_file is not None:

    st.write(
        f"**File:** {uploaded_file.name}"
    )

    if st.button("🎙️ Transcribe"):

        suffix = os.path.splitext(
            uploaded_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name

        try:

            with st.spinner(
                "Loading Whisper model..."
            ):

                whisper_model = load_whisper()

            with st.spinner(
                "Transcribing audio..."
            ):

                result = whisper_model.transcribe(
                    temp_path
                )

            transcript = result["text"].strip()

            st.session_state[
                "transcript"
            ] = transcript

            st.success(
                "Transcription completed!"
            )

        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)


# --------------------------------------------------
# Display Transcript
# --------------------------------------------------

if "transcript" in st.session_state:

    transcript = st.session_state[
        "transcript"
    ]

    st.subheader("📝 Transcript")

    st.text_area(
        "Transcription",
        transcript,
        height=300
    )


# --------------------------------------------------
# RAG Retrieval
# --------------------------------------------------

if "transcript" in st.session_state:

    st.subheader("🔎 Ask a Question")

    question = st.text_input(
        "Ask something about the uploaded audio"
    )

    if st.button("🔍 Retrieve Relevant Context"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Creating transcript chunks..."
            ):

                chunks = chunk_text(
                    transcript
                )

            with st.spinner(
                "Creating embeddings..."
            ):

                embeddings = create_embeddings(
                    chunks
                )

            with st.spinner(
                "Searching relevant information..."
            ):

                model = load_embedding_model()

                query_embedding = model.encode(
                    [question]
                )

                similarities = cosine_similarity(
                    query_embedding,
                    embeddings
                )[0]

                top_k = min(
                    3,
                    len(chunks)
                )

                top_indices = (
                    similarities
                    .argsort()[-top_k:][::-1]
                )

            st.subheader(
                "📚 Relevant Context"
            )

            for rank, idx in enumerate(
                top_indices,
                start=1
            ):

                score = similarities[idx]

                st.markdown(
                    f"**Result {rank} — "
                    f"Similarity: {score:.3f}**"
                )

                st.write(
                    chunks[idx][:500]
                )

                st.divider()