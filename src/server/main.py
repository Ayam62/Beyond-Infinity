import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from generator.generator import rag_chat, clear_chat_history
from retriever.retriever import initialize_reranker
import torch
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ============================================
# STARTUP: Load all models once at server start
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "../.env"))

if not PINECONE_API_KEY:
    raise ValueError("Please set the PINECONE_API_KEY in your .env file!")

print("🚀 Initializing API...")
print(f"📊 GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU Device: {torch.cuda.get_device_name(0)}")

# 1. Connect to Pinecone
print("📌 Connecting to Pinecone...")
INDEX_NAME = "nepali-docs-hybrid"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
print(f"✅ Connected to Pinecone index: {INDEX_NAME}")

# 2. Load Dense Embeddings (GPU accelerated if available)
print("🔤 Loading dense embeddings...")
model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
dense_embeddings = HuggingFaceEmbeddings(
    model_name="universalml/Nepali_Embedding_Model",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print(f"✅ Dense embeddings loaded on {model_kwargs['device']}")

# 3. Load BM25 Encoder
print("📝 Loading BM25 encoder...")
bm25_encoder = BM25Encoder()
bm25_path = os.path.join(BASE_DIR, "../embeddings/bm25_params.json")

if os.path.exists(bm25_path):
    bm25_encoder.load(bm25_path)
    print("✅ BM25 encoder loaded")
else:
    print(f"⚠️ Warning: BM25 parameters not found at {bm25_path}. Please generate them first!")

# 4. Pre-load Reranker Model (GPU accelerated)
print("🔄 Pre-loading reranker model...")
initialize_reranker()
print("✅ All models loaded successfully!")

# ============================================
# FastAPI App
# ============================================

app = FastAPI(title="Conversational API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Models
class ChatRequest(BaseModel):
    message: str

class SourceInfo(BaseModel):
    source_link: str
    source_type: str

class ChatResponse(BaseModel):
    reply: str
    sources: List[SourceInfo]

# Routes
@app.get("/")
def root():
    return {"message": "Conversational API is running 🚀"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    query = req.message.strip()
    if not query:
        return {"reply": "कृपया प्रश्न लेख्नुहोस्।", "sources": []}

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
        print("Error:", e)
        return {"reply": "माफ गर्नुहोस्, केही समस्या आयो।", "sources": []}

@app.post("/clear-history")
def clear_history():
    try:
        clear_chat_history()
        return {"message": "Chat history cleared successfully!"}
    except Exception as e:
        print("❌ Error clearing history:", e)
        return {"message": "⚠️ Error clearing history."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    }