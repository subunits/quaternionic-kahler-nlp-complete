"""FastAPI server for Quaternionic-Kähler Natural Language Processing."""

import sys
sys.path.insert(0, '/home/claude/quaternionic-kahler-nlp-complete')

from typing import Optional, List
from datetime import datetime
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging

from src.quaternion_nlp import QuaternionicTransformer, get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quaternionic-Kähler Natural Language Processing API",
    description="Production-ready API for quaternionic semantic embeddings",
    version="1.0.0",
    docs_url="/docs",
)

model = None
device = None


class EmbedRequest(BaseModel):
    """Request model for embedding endpoint."""
    text: str = Field(..., min_length=1, max_length=1000)
    return_magnitude: bool = Field(False)


class EmbedResponse(BaseModel):
    """Response model for embedding endpoint."""
    embedding: List[float]
    magnitude: Optional[float] = None
    timestamp: str


class SimilarityRequest(BaseModel):
    """Request model for similarity endpoint."""
    text1: str = Field(..., min_length=1)
    text2: str = Field(..., min_length=1)


class SimilarityResponse(BaseModel):
    """Response model for similarity endpoint."""
    similarity: float = Field(..., ge=0.0, le=1.0)
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model, device
    
    try:
        logger.info("Initializing model...")
        device = get_device("auto")
        
        model = QuaternionicTransformer(
            vocab_size=1000,
            embedding_dim=64,
            num_layers=2,
            num_heads=4,
        ).to(device)
        
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.get("/info")
async def get_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "name": "Quaternionic-Kähler Natural Language Processing",
        "version": "1.0.0",
        "vocab_size": model.vocab_size,
        "embedding_dim": model.embedding_dim,
        "num_layers": model.num_layers,
        "total_parameters": total_params,
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Embed text to quaternionic space."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        tokens = [ord(c) % model.vocab_size for c in request.text[:512]]
        input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embeddings, _ = model(input_ids)
            embedding = embeddings.mean(dim=1).squeeze(0).cpu()
        
        magnitude = None
        if request.return_magnitude:
            magnitude = float(torch.norm(embedding))
        
        return EmbedResponse(
            embedding=embedding.tolist(),
            magnitude=magnitude,
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.error(f"Error embedding text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    """Compute similarity between two texts."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        tokens1 = [ord(c) % model.vocab_size for c in request.text1[:512]]
        tokens2 = [ord(c) % model.vocab_size for c in request.text2[:512]]
        
        input_ids1 = torch.tensor(tokens1).unsqueeze(0).to(device)
        input_ids2 = torch.tensor(tokens2).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb1, _ = model(input_ids1)
            emb2, _ = model(input_ids2)
            
            emb1 = emb1.mean(dim=1).squeeze(0)
            emb2 = emb2.mean(dim=1).squeeze(0)
            
            similarity = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0), emb2.unsqueeze(0)
            ).item()
        
        return SimilarityResponse(
            similarity=similarity,
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Quaternionic-Kähler Natural Language Processing API",
        "version": "1.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
