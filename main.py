import os
import logging
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# OpenAI (SDK v1.x)
from openai import OpenAI

# Pinecone
from pinecone import Pinecone

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv()

app = FastAPI(title="Legal Petition Drafter API", version="1.0.0")

# CORS
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-docs")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in environment")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY missing in environment")


# Clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)


# Firebase init (optional; feedback endpoint handles absence gracefully)
firestore_client = None
firebase_creds_path = os.getenv("FIREBASE_CREDENTIALS", "firebase-credentials.json")
if os.path.exists(firebase_creds_path):
    try:
        cred = credentials.Certificate(firebase_creds_path)
        firebase_admin.initialize_app(cred)
        firestore_client = firestore.client()
    except Exception:
        firestore_client = None


# Logging
APP_DEBUG = os.getenv("APP_DEBUG", "false").lower() in ("1", "true", "yes")
logging.basicConfig(level=logging.DEBUG if APP_DEBUG else logging.INFO)
logger = logging.getLogger("legal-petition-backend")


class ChatMessage(BaseModel):
    role: str
    content: str


class PetitionRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = []


class PetitionResponse(BaseModel):
    petition: str
    conversation_id: str
    legal_context: List[str]


class FeedbackRequest(BaseModel):
    conversation_id: str
    rating: str  # 'up' | 'down'
    comment: Optional[str] = None
    petition_text: Optional[str] = None


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "Legal Petition Drafter backend is running",
        "docs": "/docs",
        "endpoints": [
            "/api/health",
            "/api/petition",
            "/api/feedback",
        ],
    }


def embed_text(text: str) -> List[float]:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding


def retrieve_legal_context(query: str, top_k: int = 5) -> List[str]:
    query_vec = embed_text(query)
    results = pinecone_index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    contexts: List[str] = []
    matches = results.matches or []
    logger.info("Pinecone retrieved %d matches", len(matches))
    for i, match in enumerate(matches):
        md = match.metadata or {}
        txt = md.get("text")
        if isinstance(txt, str):
            contexts.append(txt)
            if APP_DEBUG:
                logger.debug("Match %d score=%.4f snippet=%s", i + 1, getattr(match, "score", 0.0), txt[:160].replace("\n", " "))
    return contexts


PAKISTAN_FORMAT_PROMPT = (
    "You are a Pakistani legal drafting assistant. Draft a formal court-ready petition with: "
    "(1) Title/Cause Title, (2) Parties, (3) Jurisdiction, (4) Facts, (5) Grounds, (6) Prayer, "
    "(7) Verification, (8) List of Annexures. Follow Pakistani court style, concise, precise, formal. "
    "Cite legal provisions where relevant. Use clear headings and proper formatting."
)


def build_system_prompt(legal_context: List[str]) -> str:
    ctx_block = "\n\n".join([f"Reference {i+1}: {c}" for i, c in enumerate(legal_context)])
    prompt = (
        f"{PAKISTAN_FORMAT_PROMPT}\n\nRelevant legal context (Pakistan):\n{ctx_block}\n\n"
        "If context is insufficient, proceed with best practices but avoid fabrications."
    )
    if APP_DEBUG:
        logger.debug("System prompt preview: %s", prompt[:400].replace("\n", " "))
    return prompt


def draft_petition(user_message: str, legal_context: List[str], history: List[ChatMessage]) -> str:
    messages = []
    messages.append({"role": "system", "content": build_system_prompt(legal_context)})
    for m in history[-10:]:
        if m.role in ("user", "assistant"):
            messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": user_message})

    completion = openai_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0.2,
    )
    return completion.choices[0].message.content or ""


@app.post("/api/petition", response_model=PetitionResponse)
def create_petition(req: PetitionRequest):
    try:
        logger.info("/api/petition query: %s", (req.message or "")[:200].replace("\n", " "))
        context = retrieve_legal_context(req.message, top_k=5)
        logger.info("Context injected count: %d", len(context))
        petition_text = draft_petition(req.message, context, req.conversation_history or [])
        conv_id = str(uuid.uuid4())
        return PetitionResponse(petition=petition_text, conversation_id=conv_id, legal_context=context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate petition: {e}")


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    if firestore_client is None:
        # Accept request but indicate storage not configured
        return {"status": "ok", "stored": False, "reason": "Firestore not configured"}
    try:
        feedback_doc = {
            "conversation_id": req.conversation_id,
            "rating": req.rating,
            "comment": req.comment,
            "petition_text": req.petition_text,
            "created_at": firestore.SERVER_TIMESTAMP,
        }
        firestore_client.collection("feedback").add(feedback_doc)
        return {"status": "ok", "stored": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


