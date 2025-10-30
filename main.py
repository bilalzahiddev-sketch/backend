import os
import logging
import atexit
import time
from hashlib import sha256
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pythonjsonlogger import jsonlogger

# OpenAI (SDK v1.x)
from openai import OpenAI

# Pinecone
from pinecone import Pinecone, ServerlessSpec

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
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")  # None => default; "*" => all namespaces

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in environment")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY missing in environment")


# Clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists or create it
def ensure_pinecone_index(name: str):
    try:
        names = [i["name"] for i in pc.list_indexes()]  # type: ignore
        if name not in names:
            pc.create_index(
                name=name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
            )
    except Exception as e:
        raise RuntimeError(f"Failed ensuring Pinecone index '{name}': {e}")

ensure_pinecone_index(PINECONE_INDEX_NAME)
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
logger = logging.getLogger("legal-petition-backend")
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]
logger.setLevel(logging.DEBUG if APP_DEBUG else logging.INFO)


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


EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "30"))
CHAT_TIMEOUT = float(os.getenv("CHAT_TIMEOUT", "60"))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))

embedding_cache: Dict[str, List[float]] = {}
draft_cache: Dict[str, str] = {}


@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=0.5, min=0.5, max=4), reraise=True)
def embed_text(text: str) -> List[float]:
    key = sha256(text.encode("utf-8")).hexdigest()
    if key in embedding_cache:
        return embedding_cache[key]
    resp = openai_client.embeddings.create(model="text-embedding-3-small", input=text, timeout=EMBED_TIMEOUT)
    vec = resp.data[0].embedding
    embedding_cache[key] = vec
    return vec


def _keyword_overlap_score(text: str, query: str) -> float:
    q_terms = {t.lower() for t in query.split() if len(t) > 3}
    t_terms = {t.lower() for t in text.split() if len(t) > 3}
    if not q_terms:
        return 0.0
    inter = len(q_terms & t_terms)
    return inter / len(q_terms)


def retrieve_legal_context(query: str, top_k: int = 8) -> List[str]:
    query_vec = embed_text(query)
    contexts: List[str] = []

    # Query function for a single namespace
    def query_namespace(ns: str | None):
        res = pinecone_index.query(vector=query_vec, top_k=top_k, include_metadata=True, namespace=ns)
        return res.matches or []

    all_matches = []
    if PINECONE_NAMESPACE == "*":
        try:
            stats = pinecone_index.describe_index_stats()
            ns_map = (stats.namespaces or {})  # type: ignore
            ns_list = list(ns_map.keys()) or [None]
        except Exception:
            ns_list = [None]
        for ns in ns_list:
            all_matches.extend(query_namespace(ns))
    else:
        # specific namespace or default
        ns = PINECONE_NAMESPACE if PINECONE_NAMESPACE else None
        all_matches.extend(query_namespace(ns))

    # De-dup by (source,page,chunk)
    seen = set()
    dedup = []
    for m in all_matches:
        md = (getattr(m, "metadata", {}) or {})
        ident = (md.get("source"), md.get("page"), md.get("chunk"))
        if ident in seen:
            continue
        seen.add(ident)
        dedup.append(m)

    # Combine Pinecone score with simple keyword overlap for hybrid re-rank
    def combined_score(m):
        base = getattr(m, "score", 0.0) or 0.0
        md = (getattr(m, "metadata", {}) or {})
        txt = md.get("text") or ""
        kw = _keyword_overlap_score(txt, query)
        return 0.85 * base + 0.15 * kw

    dedup.sort(key=combined_score, reverse=True)
    top_matches = dedup[:top_k]
    logger.info("Pinecone retrieved %d matches (aggregated)", len(top_matches))
    for i, match in enumerate(top_matches):
        md = match.metadata or {}
        txt = md.get("text")
        source = md.get("source")
        page = md.get("page")
        if isinstance(txt, str):
            prefix = ""
            if source and page:
                prefix = f"[source: {source} p.{page}] "
            elif source:
                prefix = f"[source: {source}] "
            contexts.append(prefix + txt)
            if APP_DEBUG:
                logger.debug("TopMatch %d score=%.4f snippet=%s", i + 1, getattr(match, "score", 0.0), txt[:160].replace("\n", " "))
    return contexts


PAKISTAN_FORMAT_PROMPT = (
    "You are a Pakistani legal drafting assistant. Draft a formal court-ready petition with: "
    "(1) Title/Cause Title, (2) Parties, (3) Jurisdiction, (4) Facts, (5) Grounds, (6) Prayer, "
    "(7) Verification, (8) List of Annexures. Follow Pakistani court style, concise, precise, formal. "
    "Cite legal provisions where relevant. Use clear headings and proper formatting. "
    "Use bracketed citations [R1], [R2], etc., mapping strictly to the provided 'Relevant legal context' items in order. Do NOT invent sources."
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

    # Cache key based on user message + context
    cache_key = sha256((user_message + "\n".join(legal_context)).encode("utf-8")).hexdigest()
    if cache_key in draft_cache:
        return draft_cache[cache_key]

    primary = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    secondary = os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4o-mini")
    try:
        completion = openai_client.chat.completions.create(model=primary, messages=messages, temperature=0.2, timeout=CHAT_TIMEOUT)
    except Exception:
        completion = openai_client.chat.completions.create(model=secondary, messages=messages, temperature=0.2, timeout=CHAT_TIMEOUT)
    text = completion.choices[0].message.content or ""

    # Minimal validation: ensure we include a References Used section
    if "References Used" not in text:
        refs = "\n".join([f"[R{i+1}] {c[:200]}" for i, c in enumerate(legal_context)])
        text += f"\n\nReferences Used\n{refs}"

    draft_cache[cache_key] = text
    return text


@app.post("/api/petition", response_model=PetitionResponse)
def create_petition(req: PetitionRequest):
    try:
        logger.info("/api/petition query: %s", (req.message or "")[:200].replace("\n", " "))
        context = retrieve_legal_context(req.message, top_k=8)
        logger.info("Context injected count: %d", len(context))
        petition_text = draft_petition(req.message, context, req.conversation_history or [])
        conv_id = str(uuid.uuid4())
        return PetitionResponse(petition=petition_text, conversation_id=conv_id, legal_context=context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate petition: {e}")


@app.get("/api/ready")
def ready() -> Dict[str, Any]:
    # lightweight checks
    try:
        _ = pc.list_indexes()
        pinecone_ok = True
    except Exception:
        pinecone_ok = False
    try:
        # Very small embedding request for readiness
        _ = openai_client.embeddings.create(model="text-embedding-3-small", input="ping", timeout=10)
        openai_ok = True
    except Exception:
        openai_ok = False
    return {"pinecone": pinecone_ok, "openai": openai_ok, "status": (pinecone_ok and openai_ok)}


@app.on_event("shutdown")
def on_shutdown():
    # Placeholder for graceful resource cleanup
    logger.info("Shutting down backend cleanly")


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


