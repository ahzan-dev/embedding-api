FROM python:3.11-slim

WORKDIR /app

# Install dependencies without torch first
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic

# Install CPU-only PyTorch explicitly
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install sentence-transformers (will use the CPU torch we just installed)
RUN pip install --no-cache-dir sentence-transformers

# Verify it's CPU-only (will fail build if CUDA is present)
RUN python -c "import torch; assert not torch.cuda.is_available(), 'CUDA detected! Should be CPU-only'; print('✓ CPU-only PyTorch confirmed')"

# Copy application
COPY main.py .

# Download model at build time (cached in image)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('✓ Model downloaded')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]