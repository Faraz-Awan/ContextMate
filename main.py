import os
import uuid
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
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

@app.post("/upsert")
def upsert_text(
    text: str = Body(..., media_type="text/plain")
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    for chunk in chunks:
        index.upsert_records(
            namespace="__default__",
            records=[
                {
                    "_id": str(uuid.uuid4()),   # random unique ID
                    "text": chunk               # store each chunk as separate record
                }
            ]
        )

    return {"status": "success", "chunks_uploaded": len(chunks)}


@app.get("/query")
def query_db(query: str = Body(..., media_type="text/plain")):
    results = index.search(
    namespace="__default__", 
    query={
        "inputs": {"text": query}, 
        "top_k": 2
    }
    )
    return [x['fields']['text'] for x in results['result']['hits']]

@app.post("/ask-test")
def ask_llm(query: str = Body(..., media_type="text/plain")):
    system_prompt = (
        "You are an assistant that must always write your final answer in the message content, "
        "after reasoning internally. Do not leave the message content blank."
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
def ask_llm_context(query: str = Body(..., media_type="text/plain")):
    retrieved_chunks = query_db(query)
    context = "\n\n".join(retrieved_chunks)

    system_prompt = (
        "You are an assistant that answers questions "
        "based strictly on the provided context. "
        "Use citations like [S1], [S2] when referencing sources."
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
def ask_llm_context_stream(query: str = Body(..., media_type="text/plain")):
    retrieved_chunks = query_db(query)
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

    # ✅ use plain/text for curl visibility
    return StreamingResponse(stream(), media_type="text/plain")
