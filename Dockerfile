FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies (rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Python packages (rarely changes)
RUN pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    sentence-transformers \
    pydantic

# 3. Download model BEFORE copying main.py (cached!)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# 4. Copy application files LAST (changes frequently)
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]