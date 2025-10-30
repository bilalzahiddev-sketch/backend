import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-docs")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("OPENAI_API_KEY and PINECONE_API_KEY must be set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
existing = [idx["name"] for idx in pc.list_indexes()]  # type: ignore
if PINECONE_INDEX_NAME not in existing:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_REGION", "us-east-1")),
    )

index = pc.Index(PINECONE_INDEX_NAME)

def embed(text: str):
    r = client.embeddings.create(model="text-embedding-3-small", input=text)
    return r.data[0].embedding

legal_docs = [
    {
        "id": "doc1",
        "text": "Article 199 of the Constitution of Pakistan empowers High Courts to issue writs including habeas corpus, mandamus, prohibition, quo warranto and certiorari for enforcement of fundamental rights.",
    },
    {
        "id": "doc2",
        "text": "Order VII Rule 1 of the Code of Civil Procedure, 1908 prescribes the particulars required in a plaint, including names of parties, cause of action, jurisdiction facts, relief claimed, and valuation.",
    },
    {
        "id": "doc3",
        "text": "Section 9 CPC confers jurisdiction on civil courts to try all suits of a civil nature except those barred; jurisdiction may be limited by pecuniary and territorial rules.",
    },
    {
        "id": "doc4",
        "text": "Specific Relief Act provisions may be invoked for injunctions, declarations, and specific performance, subject to equitable considerations and judicial discretion.",
    },
    {
        "id": "doc5",
        "text": "Limitation Act provisions govern time-barred claims; petitions must disclose that the cause is within limitation or explain delay with sufficient cause.",
    },
]

vectors = []
for doc in legal_docs:
    vectors.append({
        "id": doc["id"],
        "values": embed(doc["text"]),
        "metadata": {"text": doc["text"]},
    })

index.upsert(vectors=vectors)
print(f"Uploaded {len(vectors)} legal documents to Pinecone index '{PINECONE_INDEX_NAME}'.")


