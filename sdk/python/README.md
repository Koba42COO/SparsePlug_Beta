# Prime-Sparse Python SDK

Official Python SDK for the [Prime-Sparse](https://prime-sparse.com) neural network optimization API.

## Installation

```bash
pip install prime-sparse
```

## Quick Start

```python
from prime_sparse import Client

# Initialize client
client = Client(api_key="your_api_key")

# One-liner optimization
result = client.optimize_file(
    "model.pt",
    output_path="optimized.safetensors",
    sparsity=0.96
)

print(f"Compressed {result.original_size_mb}MB → {result.optimized_size_mb}MB")
print(f"Compression ratio: {result.compression_ratio}x")
```

## Features

- 🚀 **96% sparsity** with <0.2% accuracy loss
- 📦 **~25x compression** ratio
- ⚡ **1.76x speedup** on CPU
- 🔄 Async support for production apps
- 🛡️ Automatic retry and error handling

## Usage

### Upload and Optimize

```python
from prime_sparse import Client

client = Client(api_key="ps_live_xxx")

# Upload model
model = client.upload_model("my_model.pt")

# Start optimization
job = client.optimize(
    model.id,
    sparsity=0.96,
    output_format="safetensors",
)

# Wait for completion
result = job.wait()

# Download optimized model
result.download("optimized.safetensors")
```

### Async Usage

```python
import asyncio
from prime_sparse import AsyncClient

async def optimize_async():
    async with AsyncClient(api_key="ps_live_xxx") as client:
        model = await client.upload_model("model.pt")
        job = await client.optimize(model.id)
        result = await job.wait()
        return result

result = asyncio.run(optimize_async())
```

### Check Usage

```python
usage = client.get_usage()

print(f"Tier: {usage.tier}")
print(f"Models: {usage.models_used}/{usage.models_limit}")
print(f"Optimizations: {usage.optimizations_used}/{usage.optimizations_limit}")
```

## Configuration

### Environment Variables

```bash
export PRIME_SPARSE_API_KEY=ps_live_xxx
export PRIME_SPARSE_API_URL=https://api.prime-sparse.com/api/v1  # Optional
```

### Client Options

```python
client = Client(
    api_key="ps_live_xxx",
    api_url="https://api.prime-sparse.com/api/v1",
    timeout=300,  # seconds
)
```

## Error Handling

```python
from prime_sparse import Client
from prime_sparse.exceptions import (
    AuthenticationError,
    RateLimitError,
    ValidationError,
    OptimizationError,
)

client = Client(api_key="ps_live_xxx")

try:
    result = client.optimize_file("model.pt")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry in {e.retry_after}s")
except ValidationError as e:
    print(f"Invalid input: {e}")
except OptimizationError as e:
    print(f"Optimization failed: {e}")
```

## API Reference

### Client Methods

| Method | Description |
|--------|-------------|
| `upload_model(path)` | Upload a model file |
| `list_models()` | List all uploaded models |
| `get_model(id)` | Get model details |
| `delete_model(id)` | Delete a model |
| `optimize(model_id, ...)` | Start optimization job |
| `get_job(id)` | Get job status |
| `list_jobs()` | List all jobs |
| `get_usage()` | Get usage summary |
| `optimize_file(path, ...)` | Upload, optimize, download in one call |

### Optimization Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sparsity` | float | 0.96 | Target sparsity (0.5-0.99) |
| `preserve_accuracy` | bool | True | Validate accuracy after optimization |
| `output_format` | str | "safetensors" | Output format (safetensors, pytorch, onnx) |
| `quantize` | bool | False | Apply INT8 quantization |

## Support

- 📚 [Documentation](https://prime-sparse.readthedocs.io)
- 💬 [Discord](https://discord.gg/prime-sparse)
- 📧 [Email](mailto:support@prime-sparse.com)
- 🐛 [Issues](https://github.com/koba42/prime-sparse/issues)

## License

MIT License - see [LICENSE](LICENSE) for details.
