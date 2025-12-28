from fastapi import FastAPI, HTTPException, Header
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List, Optional, Union
import logging
import os
from dotenv import load_dotenv  # Add this import



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Embedding API", version="1.0.0")

# API Key from environment variable
API_KEY = os.getenv("API_KEY", "your-secret-key-change-this")
print(f"Using API Key: {API_KEY}")

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

class OpenAIEmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "all-MiniLM-L6-v2"
    encoding_format: str = "float"

class OpenAIEmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key from header"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key

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
    """Health check - no auth required"""
    return {
        "status": "healthy",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": model.get_sentence_embedding_dimension()
    }

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest, api_key: str = Header(None, alias="X-API-Key")):
    """Protected endpoint - requires API key"""
    verify_api_key(api_key)
    
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
async def embed_single(text: str, api_key: str = Header(None, alias="X-API-Key")):
    """Protected endpoint - requires API key"""
    verify_api_key(api_key)
    
    try:
        embedding = model.encode([text], show_progress_bar=False)[0]
        return {
            "embedding": embedding.tolist(),
            "dimensions": len(embedding)
        }
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse)
async def embeddings_openai_compatible(
    request: OpenAIEmbeddingRequest,
    api_key: str = Header(None, alias="Authorization")
):
    """OpenAI-compatible embeddings endpoint"""
    # Handle "Bearer TOKEN" format
    if api_key and api_key.startswith("Bearer "):
        api_key = api_key[7:]
        print(f"Extracted API Key: {api_key}")
    
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Convert input to list
        input_texts = request.input if isinstance(request.input, list) else [request.input]
        
        logger.info(f"OpenAI-compatible embedding request for {len(input_texts)} texts")
        
        # Use chunking if you added it earlier
        embeddings = model.encode(
            input_texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # OpenAI response format
        return OpenAIEmbeddingResponse(
            object="list",
            data=[
                {
                    "object": "embedding",
                    "embedding": emb.tolist(),
                    "index": i
                }
                for i, emb in enumerate(embeddings)
            ],
            model=request.model,
            usage={
                "prompt_tokens": sum(len(t.split()) for t in input_texts),
                "total_tokens": sum(len(t.split()) for t in input_texts)
            }
        )
    
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
