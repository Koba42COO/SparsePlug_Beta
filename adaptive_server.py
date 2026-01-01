#!/usr/bin/env python3
"""
UPG-PAC Adaptive Server v2.0.0
Production-ready FastAPI server with:
- Slice serving (progressive sparsity streaming)
- Provenance & audit logging
- Merkle integrity exposure
- Health endpoint
- CORS (demo-friendly)
- Secure path handling
- Proper streaming headers
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from typing import Optional

from core.serving import SliceServingService
from core.provenance import audit_logger
from core.upg_pac import UPGPACReader

app = FastAPI(
    title="UPG-PAC Adaptive Server",
    description="One file. Any device. Zero tweaks.",
    version="2.0.0",
)

# Allow all origins for demo / public deployment
# In production, restrict to trusted domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
service = SliceServingService()

# Model storage directory (current dir for simplicity)
MODEL_DIR = "."

def secure_path(model_id: str) -> str:
    """Resolve and validate model path securely."""
    if ".." in model_id or model_id.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid model ID")
    
    filepath = os.path.abspath(os.path.join(MODEL_DIR, model_id))
    if not filepath.startswith(os.path.abspath(MODEL_DIR)):
        raise HTTPException(status_code=400, detail="Path traversal attempt detected")
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Model not found")
    
    return filepath


@app.get("/health")
async def health():
    """Health check endpoint for orchestrators."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "model": "UPG-PAC Adaptive Server"
    }


@app.get("/models")
async def list_models():
    """List available .upgpac models in the directory."""
    try:
        models = [
            f for f in os.listdir(MODEL_DIR)
            if f.endswith(".upgpac") and os.path.isfile(os.path.join(MODEL_DIR, f))
        ]
        return {"models": models, "count": len(models)}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to list models")


@app.get("/models/{model_id}/info")
async def model_info(model_id: str):
    """Return metadata and attribution for a model."""
    filepath = secure_path(model_id)
    
    try:
        with UPGPACReader(filepath) as reader:
            info = {
                "model_id": model_id,
                "size_mb": round(os.path.getsize(filepath) / (1024**2), 2),
                "attribution": reader.get_attribution(),
                "merkle_root": reader.get_merkle_root(),
                "tiers_available": reader.get_available_tiers(),
            }
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read model info: {str(e)}")


@app.get("/models/{model_id}/slice")
async def get_model_slice(
    model_id: str,
    sparsity: float = 1.0,
    request: Request = None
):
    """
    Stream a sliced version of the model at target sparsity.
    
    Sparsity parameter:
      - 0.99 → Extreme (IoT)
      - 0.96 → High (Mobile)
      - 0.50 → Medium
      - 0.00 → Full fidelity (Server)
    """
    if not (0.0 <= sparsity <= 1.0):
        raise HTTPException(status_code=400, detail="Sparsity must be between 0.0 and 1.0")
    
    filepath = secure_path(model_id)
    
    client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    
    try:
        # Generate streaming slice
        stream_gen = service.stream_slice(filepath, target_sparsity=sparsity)
        
        # Extract attribution and Merkle root for logging & headers
        attribution = {}
        merkle_root = "unknown"
        try:
            with UPGPACReader(filepath) as reader:
                attribution = reader.get_attribution()
                merkle_root = reader.get_merkle_root()
        except Exception:
            pass  # Non-critical — continue serving
        
        # Audit log the serve event
        audit_logger.log_event(
            event_type="slice_served",
            model_id=model_id,
            details={
                "sparsity": sparsity,
                "client_ip": client_ip,
                "user_agent": request.headers.get("User-Agent", "unknown"),
                "referer": request.headers.get("Referer", "none"),
            },
            attribution=attribution
        )
        
        # Stream response with rich headers
        return StreamingResponse(
            stream_gen,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{os.path.basename(model_id)}"',
                "X-PAC-Sparsity": f"{sparsity:.4f}",
                "X-PAC-Tier": service.get_tier_name(sparsity),
                "X-Merkle-Root": merkle_root,
                "X-Model-Size-MB": str(round(os.path.getsize(filepath) / (1024**2), 2)),
                "Cache-Control": "no-cache",
            }
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to serve slice: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "adaptive_server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
        proxy_headers=True,
    )
