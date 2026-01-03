import os
import torch
import azure.cognitiveservices.speech as speechsdk
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from pydub import AudioSegment

# Import your custom modules
from generator.generator import rag_chat, clear_chat_history
from retriever.retriever import initialize_reranker

# ============================================
# STARTUP: Configuration & Model Loading
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "../.env"))

# Azure Speech Config
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Please set the PINECONE_API_KEY in your .env file!")

print("🚀 Initializing API and Loading Models...")

# 1. Connect to Pinecone
INDEX_NAME = "nepali-docs-hybrid"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# 2. Load Dense Embeddings
model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
dense_embeddings = HuggingFaceEmbeddings(
    model_name="universalml/Nepali_Embedding_Model",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 3. Load BM25 Encoder
bm25_encoder = BM25Encoder()
bm25_path = os.path.join(BASE_DIR, "../embeddings/bm25_params.json")
if os.path.exists(bm25_path):
    bm25_encoder.load(bm25_path)

# 4. Pre-load Reranker
initialize_reranker()

# ============================================
# STT Helper Function
# ============================================

def transcribe_audio(audio_file_path: str):
    """
    Transcribes audio to text with auto-detection for Nepali and English.
    """
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        print("❌ Azure credentials missing")
        return None

    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)

    # Auto-detect language between Nepali and English
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=["ne-NP", "en-US"]
    )

    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, 
        auto_detect_source_language_config=auto_detect_source_language_config, 
        audio_config=audio_config
    )

    result = speech_recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized")
        return None
    else:
        return None

# ============================================
# FastAPI App & Models
# ============================================

app = FastAPI(title="Sahaj Conversational API", version="1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SourceInfo(BaseModel):
    source_link: str
    source_type: str

class ChatResponse(BaseModel):
    reply: str
    sources: List[SourceInfo]

class ChatRequest(BaseModel):
    message: str

# ============================================
# Core RAG Logic
# ============================================

def process_rag_request(query: str):
    """ Helper for RAG logic """
    try:
        result = rag_chat(
            index=index,
            query=query,
            dense_embeddings=dense_embeddings,
            bm25_encoder=bm25_encoder,
            alpha=0.5, 
            n_retrieval=7,
            n_generation=3
        )
        return {
            "reply": result.get("answer", "माफ गर्नुहोस्, केही समस्या आयो।"),
            "sources": result.get("sources", []),
        }
    except Exception as e:
        print(f"❌ RAG Error: {e}")
        return {"reply": "माफ गर्नुहोस्, केही समस्या आयो।", "sources": []}

# ============================================
# Endpoints
# ============================================

@app.post("/transcribe")
async def transcribe_only(file: UploadFile = File(...)):
    """
    Step 1: Convert audio to text so the user can edit it in the frontend.
    """
    raw_path = f"raw_transcribe_{file.filename}"
    wav_path = f"fixed_{file.filename}.wav"
    
    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Fix header for Azure (Converts WebM/Ogg to WAV)
        audio = AudioSegment.from_file(raw_path)
        audio.export(wav_path, format="wav")

        text = transcribe_audio(wav_path)
        return {"transcribed_text": text or ""}
    
    except Exception as e:
        print(f"Transcription error: {e}")
        return {"transcribed_text": ""}
    
    finally:
        for p in [raw_path, wav_path]:
            if os.path.exists(p):
                os.remove(p)

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Step 2: Process the text query via RAG.
    """
    query = request.message.strip()
    result = process_rag_request(query)
    return result

@app.post("/clear-history")
def clear_history():
    try:
        clear_chat_history()
        return {"message": "Chat history cleared successfully!"}
    except Exception as e:
        print(f"❌ Error clearing history: {e}")
        return {"message": "⚠️ Error clearing history."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    }