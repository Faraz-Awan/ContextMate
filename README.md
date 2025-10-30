# RAG Knowledge Assistant API

## Overview  
This RAG (Retrieval-Augmented Generation) Knowledge Assistant API enables multi-tenant access to a vector-search backed knowledge system. Users can create distinct namespaces, stream large responses through the API, and deploy easily via Docker and Google Cloud Run.  

## Key Features  
- **Multi-Tenant Namespaces**: Each user/team gets their own isolated namespace for knowledge storage.  
- **Streaming API Responses**: Receive streaming responses for long-running or large search results.  
- **Namespace Creation Endpoint**: Dynamically create new namespaces via the API.  
- **Dockerized & Cloud Ready**: Fully containerized with Docker, ready to deploy to Google Cloud Run.  

## Tech Stack  
- Python 3.x  
- FastAPI
- Pinecone
- LangChain
- OpenAI
- Docker
- Google Cloud Run

## Future Work
- Hybrid Search (Semantic + Lexical Search)
- "Delete my Namespace" button
- Content Moderation System
- Authentication & Rate Limiting
- UI/UX Polish

## Getting Started (Local)  
```bash
git clone https://github.com/Faraz-Awan/RAG.git  
cd RAG  
docker build -t rag-api .  
docker run --rm -p 8000:8000 rag-api  
# Then access at http://localhost:8000
