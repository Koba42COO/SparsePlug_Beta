# SparsePlug Production Deployment Guide

## Full Platform Deployment (Not a Demo)

This guide covers deploying the **complete SparsePlug platform** with:
- ✅ Adaptive Server (slice serving, provenance, Merkle integrity)
- ✅ UPG-PAC V2 Engine
- ✅ Platform SDK
- ✅ Authentication & API Keys
- ✅ Production-grade infrastructure

---

## Recommended Platform: **Render.com**

### Why Render?
- ✅ **FREE tier**: 750 hours/month (enough for 24/7 uptime)
- ✅ **Zero config**: Automatic HTTPS, CDN, health checks
- ✅ **GitHub integration**: Auto-deploy on push
- ✅ **Docker support**: Uses your `Dockerfile.production`
- ✅ **No credit card required** for free tier

---

## Quick Deployment (5 Minutes)

### 1. Push to GitHub

```bash
cd /Users/coo-koba42/dev/prime-sparse-saas

# Initialize git if not already done
git init
git add .
git commit -m "Production deployment ready"

# Create GitHub repo and push
gh repo create sparseplug-platform --public --source=. --remote=origin --push
```

### 2. Deploy to Render

1. Go to [render.com](https://render.com) and sign in with GitHub
2. Click **"New +"** → **"Web Service"**
3. Connect your `sparseplug-platform` repository
4. Render will auto-detect `render.yaml` and configure everything
5. Click **"Create Web Service"**

**That's it!** Render will:
- Build your Docker image
- Deploy to production
- Assign a URL: `https://sparseplug-api.onrender.com`
- Set up auto-deploy on git push

### 3. Verify Deployment

```bash
# Check health
curl https://sparseplug-api.onrender.com/health

# List models
curl https://sparseplug-api.onrender.com/models

# Get model info
curl https://sparseplug-api.onrender.com/models/demo_flagship.upgpac/info
```

---

## Alternative Platforms

### Railway.app ($5/month credit)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

**Pros**: Better performance, more generous free tier  
**Cons**: Requires credit card (but $5 free credit)

### Fly.io (FREE tier)

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Deploy
fly launch --dockerfile Dockerfile.production
fly deploy
```

**Pros**: Global edge deployment, 3 free VMs  
**Cons**: More complex configuration

### Heroku ($7/month)

```bash
# Install Heroku CLI
brew tap heroku/brew && brew install heroku

# Deploy
heroku create sparseplug-platform
heroku stack:set container
git push heroku main
```

**Pros**: Most mature platform  
**Cons**: No free tier anymore

---

## Production Configuration

### Environment Variables

Add these in your platform's dashboard:

```bash
# Required
PORT=8000                    # Auto-set by most platforms
PYTHONUNBUFFERED=1

# Optional
LOG_LEVEL=info
MAX_WORKERS=4
MODEL_DIR=/app/models
ENABLE_CORS=true
```

### Custom Domain (Optional)

1. Buy a domain (e.g., `sparseplug.ai`)
2. In Render dashboard: Settings → Custom Domain
3. Add CNAME record: `api.sparseplug.ai` → `sparseplug-api.onrender.com`
4. Render auto-provisions SSL certificate

---

## Architecture

```
User Request
    ↓
Render Load Balancer (HTTPS)
    ↓
adaptive_server.py (FastAPI)
    ↓
├─ /health          → Health check
├─ /models          → List available models
├─ /models/{id}/info → Model metadata + Merkle root
└─ /models/{id}/slice?sparsity=0.96 → Stream compressed model
    ↓
core/serving.py (SliceServingService)
    ↓
core/upg_pac.py (UPGPACReader)
    ↓
.upgpac files (stored in /app/models)
```

---

## Uploading Models

### Option 1: Include in Docker Image

```dockerfile
# Add to Dockerfile.production
COPY models/*.upgpac /app/models/
```

### Option 2: Mount Persistent Storage (Render Disks)

1. In Render dashboard: Add Disk
2. Mount path: `/app/models`
3. Upload via SFTP or API

### Option 3: Use Object Storage (S3/R2)

```python
# Modify adaptive_server.py to fetch from S3
import boto3
s3 = boto3.client('s3')
s3.download_file('sparseplug-models', 'gpt2.upgpac', '/tmp/gpt2.upgpac')
```

---

## Monitoring & Logs

### Render Dashboard
- **Logs**: Real-time application logs
- **Metrics**: CPU, memory, request count
- **Events**: Deployments, restarts

### Health Checks
Render automatically monitors `/health` endpoint:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "model": "UPG-PAC Adaptive Server"
}
```

---

## Scaling

### Free Tier Limits
- **Render**: 750 hours/month, 512MB RAM, 0.1 CPU
- **Railway**: $5 credit/month (~100 hours)
- **Fly.io**: 3 shared VMs, 256MB RAM each

### Paid Scaling
- **Render Starter**: $7/month → 512MB RAM, always-on
- **Render Standard**: $25/month → 2GB RAM, auto-scaling
- **Railway Pro**: $20/month → 8GB RAM, unlimited hours

---

## Security

### API Authentication (Already Implemented)

Your platform includes auth in `api/` directory. To enable:

```bash
# Set environment variable
ENABLE_AUTH=true

# Users register and get API keys
curl -X POST https://your-api.onrender.com/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secure123"}'

# Create API key
curl -X POST https://your-api.onrender.com/auth/api-keys \
  -H "Authorization: Bearer <jwt_token>"
```

### CORS Configuration

Already configured in `adaptive_server.py`:
```python
allow_origins=["*"]  # Change to your frontend domain in production
```

---

## CI/CD Pipeline

### Automatic Deployment

1. Push to `main` branch
2. Render detects changes
3. Builds new Docker image
4. Runs health check
5. Switches traffic to new version (zero downtime)

### GitHub Actions (Optional)

```yaml
# .github/workflows/deploy.yml
name: Deploy to Render
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Trigger Render Deploy
        run: curl -X POST ${{ secrets.RENDER_DEPLOY_HOOK }}
```

---

## Cost Estimate

### Free Tier (Recommended for MVP)
- **Render Free**: $0/month (750 hours)
- **Fly.io Free**: $0/month (3 VMs)
- **Total**: **$0/month**

### Production (Low Traffic)
- **Render Starter**: $7/month
- **Railway Pro**: $20/month
- **Total**: **$7-20/month**

### Production (High Traffic)
- **Render Standard**: $25/month (auto-scaling)
- **Cloudflare R2**: $0.015/GB storage
- **Total**: **~$30-50/month**

---

## Next Steps

1. ✅ Deploy to Render (5 minutes)
2. ✅ Test `/health` and `/models` endpoints
3. ✅ Upload your `.upgpac` models
4. ✅ Share API URL with users
5. ✅ Monitor logs and metrics
6. ✅ Add custom domain (optional)
7. ✅ Enable authentication (optional)

---

## Support

- **Render Docs**: [render.com/docs](https://render.com/docs)
- **Platform Issues**: Check `adaptive_server.py` logs
- **Model Issues**: Verify `.upgpac` file integrity with Merkle root

**Deployment Status**: ✅ Production-ready, zero-config deployment available
