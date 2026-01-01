#!/usr/bin/env python3
"""
SparsePlug Production API - Minimal Version
Simplified for cloud deployment without heavy dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

app = FastAPI(
    title="SparsePlug API",
    description="Variable Compression Platform - Production API",
    version="1.0.0",
)

# CORS for public access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.getenv("MODEL_DIR", "./models")

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0", "service": "SparsePlug API"}

@app.get("/")
async def root():
    return {
        "service": "SparsePlug - Variable Compression Platform",
        "version": "1.0.0",
        "endpoints": {"health": "/health", "models": "/models", "docs": "/docs"},
        "repository": "https://github.com/Koba42COO/SparsePlug_Beta"
    }

@app.get("/models")
async def list_models():
    try:
        if not os.path.exists(MODEL_DIR):
            return {"models": [], "count": 0}
        models = [f for f in os.listdir(MODEL_DIR) if f.endswith((".upgpac", ".bin", ".pt"))]
        return {"models": models, "count": len(models)}
    except Exception as e:
        return {"models": [], "count": 0, "error": str(e)}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("adaptive_server:app", host="0.0.0.0", port=port, workers=1)
