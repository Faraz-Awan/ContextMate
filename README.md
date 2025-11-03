# ContextMate

> Your personal AI knowledge assistant: chat with your notes, manuals, or stories.

[![Built with FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white)]()
[![Pinecone](https://img.shields.io/badge/Pinecone-00BFA5?logo=pinecone&logoColor=white)]()
[![OpenAI](https://img.shields.io/badge/-OpenAI%20API-eee?style=flat-square&logo=openai&logoColor=412991)]()
[![Deployed on Cloud Run](https://img.shields.io/badge/Google%20Cloud%20Run-4285F4?logo=googlecloud&logoColor=white)]()

**ContextMate** is a full-stack retrieval-augmented generation (RAG) web app built with **FastAPI**, **Pinecone**, and **GPT-4o**, deployed on **Google Cloud Run**.
It creates per-user semantic namespaces so you can upload any text (study notes, technical manuals, or fiction) and ask natural-language questions with streaming answers.




## Key Features  
- **Multi-Tenant Namespaces**: Each user/team gets their own isolated namespace for knowledge storage.  
- **Streaming API Responses**: Receive streaming responses for long-running or large search results.  
- **Namespace Creation Endpoint**: Dynamically create new namespaces via the API.  
- **Dockerized & Cloud Ready**: Fully containerized with Docker, ready to deploy to Google Cloud Run.  

## Tech Stack  
- Python 3.x  
- FastAPI
- Pinecone
- GPT-4o
- LangChain
- Docker
- Google Cloud Run

## Future Work
- Switch from OpenAI Completions API to Responses API
- Observability with Grafana Cloud
- Hybrid Search (Semantic + Lexical Search)
- "Delete my Namespace" button
- Content Moderation System
- Terms of Service
- Authentication & Rate Limiting
- UI/UX Polish

## Getting Started (Local)

To run **ContextMate** locally using Docker:

```bash
# 1. Clone the repository
git clone https://github.com/Faraz-Awan/ContextMate.git
cd ContextMate

# 2. Build the Docker image
docker build -t contextmate-api .

# 3. Run the container
docker run --rm -p 8000:8000 contextmate-api

# 4. Open in your browser
# → http://localhost:8000
