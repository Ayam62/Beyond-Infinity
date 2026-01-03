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

# Geospatial Imports
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

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
print("1")

# 2. Load Dense Embeddings
model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
dense_embeddings = HuggingFaceEmbeddings(
    model_name="universalml/Nepali_Embedding_Model",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("2")

# 3. Load BM25 Encoder
bm25_encoder = BM25Encoder()
bm25_path = os.path.join(BASE_DIR, "../embeddings/bm25_params.json")
if os.path.exists(bm25_path):
    bm25_encoder.load(bm25_path)
print("3")

# 4. Pre-load Reranker
initialize_reranker()
print("5")

# 5. Initialize Geolocator (OpenStreetMap)
geolocator = Nominatim(user_agent="sahaj_nepal_locator")

# ============================================
# STT Helper Function
# ============================================

def transcribe_audio(audio_file_path: str):
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        return None
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["ne-NP", "en-US"])
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, auto_detect_source_language_config=auto_detect_source_language_config, audio_config=audio_config)
    result = speech_recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    return None

# ============================================
# FastAPI Models
# ============================================

class SourceInfo(BaseModel):
    source_link: str
    source_type: str

class NearestOfficeInfo(BaseModel):
    name: str
    address: str
    latitude: float
    longitude: float
    distance_km: float

class ChatResponse(BaseModel):
    reply: str
    sources: List[SourceInfo]
    is_location_query: bool = False
    target_office: Optional[str] = None
    nearest_office: Optional[NearestOfficeInfo] = None

class ChatRequest(BaseModel):
    message: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

# ============================================
# Geospatial Logic
# ============================================

def find_nearest_location(office_name: str, user_lat: float, user_lon: float):
    try:
        search_query = f"{office_name}, Nepal"
        locations = geolocator.geocode(search_query, exactly_one=False, limit=5)
        if not locations:
            return None
        user_coords = (user_lat, user_lon)
        best_match = None
        min_dist = float('inf')
        for loc in locations:
            loc_coords = (loc.latitude, loc.longitude)
            dist = geodesic(user_coords, loc_coords).kilometers
            if dist < min_dist:
                min_dist = dist
                best_match = {
                    "name": office_name,
                    "address": loc.address,
                    "latitude": loc.latitude,
                    "longitude": loc.longitude,
                    "distance_km": round(dist, 2)
                }
        return best_match
    except Exception as e:
        print(f"🌍 Geocoding Error: {e}")
        return None

# ============================================
# Core RAG Logic
# ============================================

def process_rag_request(query: str, lat: float = None, lon: float = None):
    try:
        result = rag_chat(
            index=index, query=query, dense_embeddings=dense_embeddings,
            bm25_encoder=bm25_encoder, alpha=0.5, n_retrieval=7, n_generation=3
        )
        answer = result.get("answer", "माफ गर्नुहोस्, केही समस्या आयो।")
        is_loc = result.get("is_location_query", False)
        office = result.get("target_office")
        
        nearest_data = None
        if is_loc and office and lat is not None and lon is not None:
            nearest_data = find_nearest_location(office, lat, lon)
            # We don't append the address to the string anymore because we will show the map
            if nearest_data:
                answer += f"\n\n📍 तपाईंको नजिकको कार्यालय तलको नक्सामा हेर्नुहोस्:"

        return {
            "reply": answer,
            "sources": result.get("sources", []),
            "is_location_query": is_loc,
            "target_office": office,
            "nearest_office": nearest_data
        }
    except Exception as e:
        print(f"❌ RAG Error: {e}")
        return {"reply": "माफ गर्नुहोस्, केही समस्या आयो।", "sources": [], "is_location_query": False}

# ============================================
# Endpoints
# ============================================

app = FastAPI(title="Sahaj Conversational API", version="1.4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/transcribe")
async def transcribe_only(file: UploadFile = File(...)):
    raw_path, wav_path = f"raw_{file.filename}", f"fixed_{file.filename}.wav"
    with open(raw_path, "wb") as buffer: buffer.write(await file.read())
    try:
        audio = AudioSegment.from_file(raw_path)
        audio.export(wav_path, format="wav")
        text = transcribe_audio(wav_path)
        return {"transcribed_text": text or ""}
    except Exception as e:
        return {"transcribed_text": ""}
    finally:
        for p in [raw_path, wav_path]:
            if os.path.exists(p): os.remove(p)

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return process_rag_request(request.message.strip(), request.latitude, request.longitude)

@app.post("/clear-history")
def clear_history():
    clear_chat_history()
    return {"message": "Success"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}