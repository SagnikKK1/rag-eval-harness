# Demo image: builds the index on first start, then serves the FastAPI app.
FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf_cache

COPY requirements.txt ./
RUN pip install -r requirements.txt && pip install fastapi "uvicorn[standard]"

COPY rag/ rag/
COPY eval/ eval/
COPY scripts/ scripts/
COPY app/ app/
COPY data/raw/ data/raw/

EXPOSE 8000
# Build FAISS + BM25 (downloads the embedding model once), then serve.
CMD ["sh", "-c", "python scripts/build_index.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
