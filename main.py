import os
import uuid
import hashlib
from fastapi import FastAPI, HTTPException, Body, Request, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not all([PINECONE_API_KEY, PINECONE_INDEX_HOST, OPENROUTER_API_KEY]):
    raise RuntimeError("Missing one or more required environment variables.")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)

app = FastAPI()

# Allow CORS only for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",  # your local Live Server origin
        "http://localhost:5500",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

def santize_namespace(ns: str) -> str:
    """
    Pinecone namespaces should be short and URL-safe.
    We hash the incoming user id to a compact, stable token.
    """
    # Normalize and hash to 16 hex chars (64 bits) to keep namespaces short & uniform
    h = hashlib.sha256(ns.encode("utf-8")).hexdigest()[:16]
    return f"user-{h}"

def get_namespace(request: Request) -> str:
    user_id = request.headers.get("X-User-ID")
    if not user_id or not isinstance(user_id, str) or len(user_id) > 200:
        raise HTTPException(status_code=400, detail="Missing or invalid X-User-ID")
    return santize_namespace(user_id)

@app.get("/", include_in_schema=False)
def serve_root():
    return FileResponse("static/index.html")

@app.get("/whoami")
def whoami(request: Request):
    """Returns the current namespace and a short display id for UI."""
    ns = get_namespace(request)
    return {"namespace": ns, "display_id": ns.replace("user-", "")[:8]}

@app.post("/upsert")
def upsert_text(
    request: Request,
    text: str = Body(..., media_type="text/plain")
):
    namespace = get_namespace(request)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    for chunk in chunks:
        index.upsert_records(
            namespace=namespace,
            records=[
                {
                    "_id": str(uuid.uuid4()),   # random unique ID
                    "text": chunk               # store each chunk as separate record
                }
            ]
        )

    return {"status": "success", "namespace": namespace, "chunks_uploaded": len(chunks)}


@app.get("/query")
def query_db_get(request: Request, query: str = Query(..., min_length=1)):
    namespace = get_namespace(request)
    results = index.search(
        namespace=namespace, 
        query={"inputs": {"text": query}, "top_k": 10}
    )
    return [x['fields']['text'] for x in results['result']['hits']]


@app.post("/query")
def query_db_post(request: Request, query: str = Body(..., media_type="text/plain")):
    return JSONResponse(content=query_db_get(request, query))

@app.post("/ask-test")
def ask_llm(request: Request, query: str = Body(..., media_type="text/plain")):
    system_prompt = (
        "You are an assistant that must always write your final answer in the message content, "
        "after reasoning internally. Do not leave the message content blank. If you are not sure,"
        "state that you don't know the answer."
    )

    user_prompt = f"Question: {query}"

    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=5120
    )
    return completion.choices[0].message.content

@app.post("/ask-context")
def ask_llm_context(request: Request, query: str = Body(..., media_type="text/plain")):
    retrieved_chunks = query_db_get(request, query)
    context = "\n\n".join(retrieved_chunks)

    system_prompt = (
        "You are a concise and factual assistant."
        "Always provide clear, well-structured answers to the user’s questions." 
        "If you are not confident that your answer is correct, or if the information is not verifiable, say “I’m not sure about that” instead of guessing." 
        "Do not make up facts, people, events, numbers, or citations." 
        "Base your reasoning only on general, reliable knowledge, not speculation or imagination." 
        "Keep your answers short and focused unless the user explicitly asks for detail."
        )

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=512
    )

    return completion.choices[0].message.content

@app.post("/ask-context-stream")
def ask_llm_context_stream(request: Request, query: str = Body(..., media_type="text/plain")):
    retrieved_chunks = query_db_get(request, query)
    context = "\n\n".join(retrieved_chunks)

    system_prompt = (
        "You are an assistant that answers questions strictly based on the provided context. "
        "If the context does not contain the answer, say 'I don't know'. "
        "Use citations like [S1], [S2] when referencing sources."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    def stream():
        completion_stream = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
            stream=True,
        )

        for chunk in completion_stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            text_piece = delta.content or ""
            if text_piece:
                yield text_piece.encode("utf-8")
            finish_reason = chunk.choices[0].finish_reason
            if finish_reason is not None:
                break

    return StreamingResponse(stream(), media_type="text/plain")