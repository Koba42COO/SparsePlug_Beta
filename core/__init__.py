"""
Prime-Sparse Core
=================

GGUF-inspired optimizations for extreme model compression.

Modules:
- quantization: 4-bit sparse quantization (200× compression)
- block_sparse: Cache-efficient block operations
- mmap_loader: Memory-mapped instant loading
- simd_kernels: Vectorized sparse operations
- upg_pac: Unified format combining all optimizations
"""

from .quantization import (
    QuantFormat,
    Q4_0, Q4_K, Q8_0, Q2_K,
    quantize_sparse_tensor,
    dequantize_sparse_blocks,
    QuantizedSparseLinear,
    SparseQuantizedBlock,
    analyze_compression
)

from .block_sparse import (
    BLOCK_SIZE,
    BlockSparseWeight,
    BlockSparseLinear,
    BlockSparseMLP,
    block_sparse_matmul
)

from .mmap_loader import (
    UPGPACWriter as MMapWriter,
    UPGPACReader as MMapReader,
    MemoryMappedModel,
    save_model_upgpac as save_mmap,
    load_state_dict_mmap
)

from .simd_kernels import (
    sparse_matmul_csr,
    fused_sparse_gelu_linear,
    sparse_linear,
    HAS_NUMBA
)

from .upg_pac import (
    UPGPACWriter,
    UPGPACReader,
    UPGPACModel,
    convert_model_to_upg_pac,
    UPG_PAC_VERSION
)

from .formula import Formula
from .audio_upg import UPGAudioEngine
from .vision_upg import UPGVisionEngine


__version__ = "2.0.0"
__all__ = [
    # Quantization
    'QuantFormat', 'Q4_0', 'Q4_K', 'Q8_0', 'Q2_K',
    'quantize_sparse_tensor', 'dequantize_sparse_blocks',
    'QuantizedSparseLinear', 'analyze_compression',
    
    # Block Sparse
    'BLOCK_SIZE', 'BlockSparseWeight', 'BlockSparseLinear',
    'BlockSparseMLP', 'block_sparse_matmul',
    
    # Memory Mapping
    'MMapWriter', 'MMapReader', 'MemoryMappedModel',
    'save_mmap', 'load_state_dict_mmap',
    
    # SIMD Kernels
    'sparse_matmul_csr', 'fused_sparse_gelu_linear',
    'sparse_linear', 'HAS_NUMBA',
    
    # UPG-PAC Format
    'UPGPACWriter', 'UPGPACReader', 'UPGPACModel',
    'convert_model_to_upg_pac', 'UPG_PAC_VERSION',

    # The Formula (Universal Interface)
    'Formula', 'UPGAudioEngine', 'UPGVisionEngine'
]
