# RAG

## Running with Docker

Build the image:

```bash
docker build -t rag-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 rag-api
```

Once running, the API is available at `http://localhost:8000`.
