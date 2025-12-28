from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Embedding API", version="1.0.0")

# Load model on startup
logger.info("Loading sentence transformer model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
logger.info(f"Model loaded. Dimensions: {model.get_sentence_embedding_dimension()}")

class EmbedRequest(BaseModel):
    texts: List[str]
    batch_size: int = 32

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    count: int

@app.get("/")
async def root():
    return {
        "service": "Embedding API",
        "model": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "status": "ready"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "sentence-transformers/all-MiniLM-L6-v2",  # Fixed
        "dimensions": model.get_sentence_embedding_dimension()
    }

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="texts cannot be empty")
        
        if len(request.texts) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 texts per request")
        
        logger.info(f"Embedding {len(request.texts)} texts")
        
        embeddings = model.encode(
            request.texts,
            batch_size=request.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            dimensions=model.get_sentence_embedding_dimension(),
            count=len(embeddings)
        )
    
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/single")
async def embed_single(text: str):
    """Quick endpoint for single text embedding"""
    try:
        embedding = model.encode([text], show_progress_bar=False)[0]
        return {
            "embedding": embedding.tolist(),
            "dimensions": len(embedding)
        }
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))