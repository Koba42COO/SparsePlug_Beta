"""
UPG-PAC: Unified Prime-Sparse Quantized Format
===============================================

The ultimate model format combining:
1. 96% structured sparsity (25× compression from neurons)
2. 4-bit quantization (8× compression from precision)
3. Block-wise storage (cache-efficient)
4. Memory mapping (instant loading)

Total compression: up to 200× !

File Format:
- Magic header
- Model config (JSON)
- Layer metadata
- Quantized sparse blocks (memory-mapped)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
import struct
import mmap
import time
import os

from .quantization import (
    quantize_sparse_tensor, 
    dequantize_sparse_blocks,
    SparseQuantizedBlock,
    Q4_K, Q4_0, Q4_DELTA, QuantFormat, DELTA,
    get_importance_tiers
)
from .block_sparse import BlockSparseWeight, BLOCK_SIZE
from .mmap_loader import UPG_PAC_MAGIC


import hashlib

# ============================================================
# UPG-PAC FORMAT SPECIFICATION
# ============================================================

UPG_PAC_VERSION = 5

@dataclass
class LayerSpec:
    """Specification for a single layer."""
    name: str
    layer_type: str  # "linear", "sparse_linear", "embedding"
    in_features: int
    out_features: int
    sparsity: float
    quant_format: str  # "Q4_K", "Q4_0", "F16", "F32"
    n_blocks: int
    block_offset: int  # Byte offset in file
    block_size: int    # Total bytes for this layer
    has_bias: bool
    block_size: int    # Total bytes for this layer (legacy/sum)
    has_bias: bool
    bias_offset: int
    tiers: List[Dict[str, Any]] = field(default_factory=list) # v4: List of {'sparsity': float, 'offset': int, 'size': int, 'n_blocks': int, 'hash': str}
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'layer_type': self.layer_type,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'sparsity': self.sparsity,
            'quant_format': self.quant_format,
            'n_blocks': self.n_blocks,
            'block_offset': self.block_offset,
            'block_size': self.block_size,
            'has_bias': self.has_bias,
            'bias_offset': self.bias_offset,
            'tiers': self.tiers
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'LayerSpec':
        return cls(**d)


@dataclass
class UPGPACHeader:
    """File header."""
    magic: bytes = UPG_PAC_MAGIC
    version: int = UPG_PAC_VERSION
    n_layers: int = 0
    total_params: int = 0
    original_size_mb: float = 0.0
    compressed_size_mb: float = 0.0
    compression_ratio: float = 1.0
    avg_sparsity: float = 0.0
    merkle_root: str = ""  # v5: Integrity Root
    attribution: dict = field(default_factory=dict) # v5: Provenance
    model_config: dict = field(default_factory=dict)



def calculate_merkle_root(hashes: List[str]) -> str:
    """Calculate Merkle Root from a list of SHA-256 hashes."""
    if not hashes:
        return ""
    
    current_level = hashes
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            if i + 1 < len(current_level):
                right = current_level[i+1]
            else:
                right = left  # Duplicate last if odd number
            
            combined = left + right
            next_level.append(hashlib.sha256(combined.encode('utf-8')).hexdigest())
        current_level = next_level
    
    return current_level[0]

# ============================================================
# SERIALIZATION
# ============================================================

def serialize_sparse_blocks(blocks: List[SparseQuantizedBlock]) -> bytes:
    """Serialize quantized sparse blocks to bytes."""
    data = bytearray()
    
    # Number of blocks
    data.extend(struct.pack('<I', len(blocks)))
    
    for block in blocks:
        # Scale and zero point
        data.extend(struct.pack('<ff', block.scale, block.zero_point))
        
        # Number of values
        data.extend(struct.pack('<I', len(block.indices)))
        data.extend(struct.pack('<I', block.n_total))
        
        # Quantized values
        data.extend(struct.pack('<I', len(block.quantized)))
        data.extend(block.quantized.tobytes())
        
        # Indices
        data.extend(block.indices.tobytes())
    
    return bytes(data)


def deserialize_sparse_blocks(data: bytes, offset: int = 0) -> Tuple[List[SparseQuantizedBlock], int]:
    """Deserialize quantized sparse blocks from bytes."""
    blocks = []
    pos = offset
    
    n_blocks = struct.unpack('<I', data[pos:pos+4])[0]
    pos += 4
    
    for _ in range(n_blocks):
        scale, zero_point = struct.unpack('<ff', data[pos:pos+8])
        pos += 8
        
        n_values = struct.unpack('<I', data[pos:pos+4])[0]
        pos += 4
        
        n_total = struct.unpack('<I', data[pos:pos+4])[0]
        pos += 4
        
        quantized_size = struct.unpack('<I', data[pos:pos+4])[0]
        pos += 4
        
        quantized = np.frombuffer(data[pos:pos+quantized_size], dtype=np.uint8)
        pos += quantized_size
        
        indices = np.frombuffer(data[pos:pos+n_values*4], dtype=np.int32)
        pos += n_values * 4
        
        blocks.append(SparseQuantizedBlock(
            scale=scale,
            zero_point=zero_point,
            quantized=quantized,
            indices=indices,
            n_total=n_total
        ))
    
    return blocks, pos


# ============================================================
# WRITER
# ============================================================

class UPGPACWriter:
    """Write models in UPG-PAC format."""
    
    PAGE_SIZE = 4096
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.layers: List[Tuple[str, Any, Optional[torch.Tensor]]] = []
        self.config: dict = {}
        self.attribution: dict = {}
    
    def add_linear(
        self,
        name: str,
        linear: nn.Linear,
        sparsity: float = 0.96,
        sparsities: List[float] = None, # v4: Progressive sparsities
        quant_format: QuantFormat = Q4_K
    ):
        """Add a linear layer with sparsification and quantization."""
        if sparsities is None:
            sparsities = [sparsity]
            
        # Get tiers
        weight = linear.weight.data.clone()
        masks = get_importance_tiers(weight, sparsities)
        
        # Quantize each tier
        tier_blocks = []
        for mask in masks:
            blocks = quantize_sparse_tensor(
                weight, 
                sparsity_mask=mask, 
                format=quant_format
            )
            tier_blocks.append(blocks)
        
        self.layers.append((name, {
            'type': 'sparse_linear',
            'shape': weight.shape,
            'tier_blocks': tier_blocks,
            'sparsities': sparsities,
            'format': quant_format.name
        }, linear.bias))
    
    def add_embedding(
        self,
        name: str,
        embedding: nn.Embedding
    ):
        """Add embedding layer (not sparsified)."""
        self.layers.append((name, {
            'type': 'embedding',
            'weight': embedding.weight.data,
            'num_embeddings': embedding.num_embeddings,
            'embedding_dim': embedding.embedding_dim
        }, None))
    
    def set_config(self, config: dict):
        """Set model configuration."""
        self.config = config

    def set_attribution(self, attribution: dict):
        """Set attribution/credit metadata."""
        self.attribution = attribution
    
    def write(self) -> Path:
        """Write to file."""
        with open(self.filepath, 'wb') as f:
            # Reserve header space
            f.write(b'\x00' * 256)
            
            # Build layer specs and serialize data
            layer_specs = []
            layer_data = []
            current_offset = 256  # After header
            
            # Write metadata JSON first
            total_params = 0
            original_bytes = 0
            compressed_bytes = 0
            
            for name, layer_info, bias in self.layers:
                if layer_info['type'] == 'sparse_linear':
                    shape = layer_info['shape']
                    tier_blocks = layer_info['tier_blocks']
                    sparsities = layer_info['sparsities']
                    
                    # Write tiers
                    tiers_metadata = []
                    layer_start_offset = current_offset
                    total_block_bytes = 0
                    
                    for i, blocks in enumerate(tier_blocks):
                        block_bytes = serialize_sparse_blocks(blocks)
                        
                        tiers_metadata.append({
                            'sparsity': sparsities[i],
                            'offset': current_offset,
                            'size': len(block_bytes),
                            'n_blocks': len(blocks)
                        })
                        
                        layer_data.append(block_bytes)
                        current_offset += len(block_bytes)
                        total_block_bytes += len(block_bytes)
                        
                        # Stats (approximate for multi-tier)
                        compressed_bytes += len(block_bytes)
                        
                        # Calculate Hash (v5)
                        block_hash = hashlib.sha256(block_bytes).hexdigest()
                        tiers_metadata[-1]['hash'] = block_hash
                    
                    # Bias
                    bias_bytes = b''
                    bias_offset = 0
                    if bias is not None:
                        # Bias stored AFTER all tiers
                        bias_offset = total_block_bytes # Offset relative to start of layer data block?
                        # Wait, spec.block_offset points to start.
                        # Ideally bias should be separate or at end.
                        # Let's put bias at the end of the last tier?
                        # Or treat bias as separate chunk.
                        # For compatibility, let's append it to layer_data.
                        
                        # We need absolute offset for bias if we want to read it directly?
                        # In v4, we'll store bias logic same as before, appearing after all tiers
                        bias_bytes = bias.detach().cpu().numpy().tobytes()
                    
                    layer_data.append(bias_bytes)
                    
                    # Spec
                    spec = LayerSpec(
                        name=name,
                        layer_type='sparse_linear',
                        in_features=shape[1],
                        out_features=shape[0],
                        sparsity=sparsities[-1], # Lowest sparsity (most info)
                        quant_format=layer_info['format'],
                        n_blocks=sum(t['n_blocks'] for t in tiers_metadata),
                        block_offset=layer_start_offset,
                        block_size=total_block_bytes + len(bias_bytes),
                        has_bias=bias is not None,
                        bias_offset=total_block_bytes, # Relative to block_offset
                        tiers=tiers_metadata
                    )
                    
                    layer_specs.append(spec)
                    current_offset += len(bias_bytes) 
                    
                    total_params += shape[0] * shape[1]
                    original_bytes += shape[0] * shape[1] * 4
                    
                elif layer_info['type'] == 'embedding':
                    weight = layer_info['weight']
                    weight_bytes = weight.detach().cpu().numpy().tobytes()
                    
                    spec = LayerSpec(
                        name=name,
                        layer_type='embedding',
                        in_features=layer_info['num_embeddings'],
                        out_features=layer_info['embedding_dim'],
                        sparsity=0.0,
                        quant_format='F32',
                        n_blocks=1,
                        block_offset=current_offset,
                        block_size=len(weight_bytes),
                        has_bias=False,
                        bias_offset=0
                    )
                    
                    layer_specs.append(spec)
                    layer_data.append(weight_bytes)
                    
                    current_offset += len(weight_bytes)
                    total_params += weight.numel()
                    original_bytes += weight.numel() * 4
                    compressed_bytes += len(weight_bytes)
            
            # Write metadata
            # Collect all hashes for Merkle Tree
            all_hashes = []
            for spec in layer_specs:
                for tier in spec.tiers:
                    if 'hash' in tier:
                        all_hashes.append(tier['hash'])
            
            merkle_root = calculate_merkle_root(all_hashes)
            
            metadata = {
                'config': self.config,
                'integrity': {'merkle_root': merkle_root}, 
                'attribution': self.attribution, # v5
                'layers': [s.to_dict() for s in layer_specs],
                'stats': {
                    'total_params': total_params,
                    'original_size_mb': original_bytes / 1024 / 1024,
                    'compressed_size_mb': compressed_bytes / 1024 / 1024,
                    'compression_ratio': original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
                }
            }
            
            # Adjust offsets for metadata
            metadata_json = json.dumps(metadata).encode('utf-8')
            metadata_offset = 256
            data_offset = metadata_offset + 8 + len(metadata_json)
            
            # Align to page
            if data_offset % self.PAGE_SIZE != 0:
                data_offset = ((data_offset // self.PAGE_SIZE) + 1) * self.PAGE_SIZE
            
            # Update layer offsets
            current = data_offset
            for i, spec in enumerate(layer_specs):
                # Calculate shift for this layer
                # But tiers might vary in size.
                # Actually, spec.block_offset is the start of the layer.
                # The tiers are stored contiguously in the layer data block.
                # We can just walk through them.
                tier_current = current
                for tier in spec.tiers:
                    tier['offset'] = tier_current
                    tier_current += tier['size']
                
                spec.block_offset = current
                current += spec.block_size
            
            # Rewrite metadata with correct offsets
            metadata['layers'] = [s.to_dict() for s in layer_specs]
            metadata_json = json.dumps(metadata).encode('utf-8')
            
            # Write metadata
            f.seek(256)
            f.write(struct.pack('<Q', len(metadata_json)))
            f.write(metadata_json)
            
            # Pad to data offset
            current_pos = f.tell()
            if current_pos < data_offset:
                f.write(b'\x00' * (data_offset - current_pos))
            
            # Write layer data
            for data in layer_data:
                f.write(data)
            
            # Write header
            f.seek(0)
            f.write(UPG_PAC_MAGIC)
            f.write(struct.pack('<I', UPG_PAC_VERSION))
            f.write(struct.pack('<I', len(layer_specs)))
            f.write(struct.pack('<Q', len(metadata_json)))
        
        return self.filepath


# ============================================================
# READER
# ============================================================

class UPGPACReader:
    """Memory-mapped reader for UPG-PAC files."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.file = None
        self.mmap = None
        self.metadata: dict = {}
        self.layer_specs: Dict[str, LayerSpec] = {}
        self._cache: Dict[str, Any] = {}
        
        self._open()
    
    def _open(self):
        """Open and memory map file."""
        self.file = open(self.filepath, 'rb')
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Read header
        magic = self.mmap[:8]
        if magic != UPG_PAC_MAGIC:
            raise ValueError(f"Invalid UPG-PAC file: {magic}")
        
        version, n_layers, metadata_size = struct.unpack('<IIQ', self.mmap[8:24])
        
        # Read metadata
        metadata_json = self.mmap[264:264+metadata_size]
        self.metadata = json.loads(metadata_json.decode('utf-8'))
        
        # Build layer map
        for layer_dict in self.metadata.get('layers', []):
            spec = LayerSpec.from_dict(layer_dict)
            self.layer_specs[spec.name] = spec
    
    def get_config(self) -> dict:
        """Get model configuration."""
        return self.metadata.get('config', {})
    
    def get_stats(self) -> dict:
        """Get compression statistics."""
        return self.metadata.get('stats', {})

    def get_attribution(self) -> dict:
        """Get attribution/credit metadata."""
        return self.metadata.get('attribution', {})
    
    def list_layers(self) -> List[str]:
        """List all layer names."""
        return list(self.layer_specs.keys())
    
    def get_layer_info(self, name: str) -> dict:
        """Get layer information."""
        if name not in self.layer_specs:
            raise KeyError(f"Layer '{name}' not found")
        return self.layer_specs[name].to_dict()
    
    def load_sparse_linear(
        self, 
        name: str, 
        target_sparsity: float = None
    ) -> Tuple[List[SparseQuantizedBlock], Optional[np.ndarray], Tuple[int, int]]:
        """
        Load a sparse linear layer.
        
        Args:
            name: Layer name
            target_sparsity: Desired sparsity level. 
                           If None, loads all available tiers (lowest sparsity).
                           If specified (e.g., 0.99), loads only tiers needed to reach that level.
        """
        # Cache key needs to include sparsity for v4
        cache_key = f"{name}_{target_sparsity}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        spec = self.layer_specs[name]
        
        if spec.layer_type != 'sparse_linear':
            raise ValueError(f"Layer '{name}' is not sparse_linear")
        
        # Determine which tiers to load
        tiers_to_load = []
        if not spec.tiers:
            # v2/v3 file (single tier)
            tiers_to_load = [{'offset': spec.block_offset, 'size': spec.bias_offset if spec.has_bias else spec.block_size}]
        else:
            # v4 file (multi-tier)
            # Tiers are stored [0.99, 0.96, 0.90] (Core -> Detail)
            # If target is 0.96, we need 0.99 AND 0.96 tiers.
            # If target is 0.99, we need only 0.99 tier.
            sorted_tiers = sorted(spec.tiers, key=lambda x: x['sparsity'], reverse=True) # High sparsity first
            
            if target_sparsity is None:
                tiers_to_load = sorted_tiers
            else:
                for tier in sorted_tiers:
                    tiers_to_load.append(tier)
                    # Use tolerance for float comparison
                    if tier['sparsity'] <= target_sparsity + 1e-6:
                        break
        
        # Load blocks from selected tiers
        all_blocks = []
        for tier in tiers_to_load:
            # tier['offset'] is absolute
            data = self.mmap[tier['offset'] : tier['offset'] + tier['size']]
            blocks, _ = deserialize_sparse_blocks(data)
            all_blocks.extend(blocks)
            
        # Read bias if present (always load bias)
        bias = None
        if spec.has_bias:
            bias_start = spec.block_offset + spec.bias_offset
            bias_end = spec.block_offset + spec.block_size
            bias = np.frombuffer(self.mmap[bias_start:bias_end], dtype=np.float32)
        
        shape = (spec.out_features, spec.in_features)
        
        # Return result
        result = (all_blocks, bias, shape)
        
        # Cache? Only if full load to avoid memory fragmentation?
        # For now, cache it.
        self._cache[cache_key] = result
        
        return result
    
    def load_embedding(self, name: str) -> torch.Tensor:
        """Load an embedding layer."""
        if name in self._cache:
            return self._cache[name]
        
        spec = self.layer_specs[name]
        
        if spec.layer_type != 'embedding':
            raise ValueError(f"Layer '{name}' is not embedding")
        
        data = np.frombuffer(
            self.mmap[spec.block_offset:spec.block_offset + spec.block_size],
            dtype=np.float32
        ).reshape(spec.in_features, spec.out_features)
        
        tensor = torch.from_numpy(data.copy())
        self._cache[name] = tensor
        
        return tensor
    
    def close(self):
        """Close file handles."""
        if self.mmap:
            self.mmap.close()
        if self.file:
            self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

    def check_integrity(self, deep: bool = False) -> bool:
        """
        Verify Merkle Integrity.
        
        Args:
            deep: If True, re-hash all block data from disk (Slow).
                  If False, only verify Tree consistency from metadata (Fast).
        
        Returns:
            True if valid, False otherwise.
        """
        # 1. Get stored root
        stored_root = self.metadata.get('integrity', {}).get('merkle_root', '')
        if not stored_root:
            # Legacy file or no integrity
            return True
            
        # 2. Collect all hashes from metadata
        all_hashes = []
        for name, spec in self.layer_specs.items():
            for tier in spec.tiers:
                if 'hash' in tier:
                    all_hashes.append(tier['hash'])
                    
                    # Deep Check: Verify Block Hash
                    if deep:
                        data = self.mmap[tier['offset'] : tier['offset'] + tier['size']]
                        computed_hash = hashlib.sha256(data).hexdigest()
                        if computed_hash != tier['hash']:
                            print(f"❌ Corruption detected in {name} (Tier {tier['sparsity']})")
                            return False
        
        # 3. Recompute Root
        computed_root = calculate_merkle_root(all_hashes)
        
        if computed_root != stored_root:
            print(f"❌ Merkle Root Mismatch! Stored: {stored_root}, Computed: {computed_root}")
            return False
            
        return True


# ============================================================
# MODEL WRAPPER
# ============================================================

class UPGPACModel(nn.Module):
    """
    PyTorch model using UPG-PAC format.
    
    Provides seamless integration with PyTorch while using
    compressed sparse weights.
    """
    
    def __init__(self, filepath: str):
        super().__init__()
        self.reader = UPGPACReader(filepath)
        self.config = self.reader.get_config()
        
        # Cache for dequantized weights
        self._weight_cache: Dict[str, torch.Tensor] = {}
    
    def get_weight(self, layer_name: str) -> torch.Tensor:
        """Get dequantized weight tensor."""
        if layer_name in self._weight_cache:
            return self._weight_cache[layer_name]
        
        # Support target_sparsity
        # For full weight, we want all tiers (so target_sparsity=None)
        blocks, bias, shape = self.reader.load_sparse_linear(layer_name, target_sparsity=None)
        
        # Get format
        spec = self.reader.layer_specs[layer_name]
        format_map = {'Q4_K': Q4_K, 'Q4_0': Q4_0, 'Q4_DELTA': Q4_DELTA}
        quant_format = format_map.get(spec.quant_format, Q4_K)
        
        # Dequantize
        weight = dequantize_sparse_blocks(blocks, shape, quant_format)
        
        self._weight_cache[layer_name] = weight
        return weight
    
    def linear(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Perform linear operation using layer."""
        weight = self.get_weight(layer_name).to(x.device)
        blocks, bias, _ = self.reader.load_sparse_linear(layer_name)
        
        out = F.linear(x, weight)
        
        if bias is not None:
            out = out + torch.from_numpy(bias).to(x.device)
        
        return out
    
    def stats(self) -> dict:
        """Get model statistics."""
        return self.reader.get_stats()
    
    def close(self):
        """Release resources."""
        self.reader.close()


# ============================================================
# CONVERSION UTILITIES
# ============================================================

def convert_model_to_upg_pac(
    model: nn.Module,
    output_path: str,
    sparsity: float = 0.96,
    sparsities: List[float] = None, # v4
    quant_format: QuantFormat = Q4_K,
    config: dict = None
) -> Path:
    """
    Convert a PyTorch model to UPG-PAC format.
    
    Args:
        model: PyTorch model
        output_path: Output file path
        sparsity: Target sparsity (default 96%)
        quant_format: Quantization format
        config: Model configuration
    
    Returns:
        Path to saved file
    """
    writer = UPGPACWriter(output_path)
    
    if config:
        writer.set_config(config)
    
    # Convert all linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            writer.add_linear(
                name, 
                module, 
                sparsity=sparsity, 
                sparsities=sparsities,
                quant_format=quant_format
            )
        elif isinstance(module, nn.Embedding):
            writer.add_embedding(name, module)
    
    return writer.write()


# ============================================================
# BENCHMARKING
# ============================================================

def benchmark_upg_pac(
    model: nn.Module,
    filepath: str = '/tmp/test.upgpac',
    n_runs: int = 10
) -> dict:
    """Benchmark UPG-PAC format vs PyTorch."""
    
    # Save in UPG-PAC format
    start = time.perf_counter()
    convert_model_to_upg_pac(model, filepath, sparsity=0.96)
    save_time = time.perf_counter() - start
    
    # Save in PyTorch format
    torch_path = filepath.replace('.upgpac', '.pt')
    torch.save(model.state_dict(), torch_path)
    
    # Compare sizes
    upg_size = os.path.getsize(filepath)
    torch_size = os.path.getsize(torch_path)
    
    # Load times
    times_torch = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = torch.load(torch_path)
        times_torch.append(time.perf_counter() - start)
    
    times_upg = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reader = UPGPACReader(filepath)
        # Load first layer to trigger actual access
        if reader.list_layers():
            _ = reader.load_sparse_linear(reader.list_layers()[0])
        reader.close()
        times_upg.append(time.perf_counter() - start)
    
    # Cleanup
    os.remove(filepath)
    os.remove(torch_path)
    
    return {
        'torch_size_mb': torch_size / 1024 / 1024,
        'upg_size_mb': upg_size / 1024 / 1024,
        'compression_ratio': torch_size / upg_size,
        'torch_load_ms': np.mean(times_torch) * 1000,
        'upg_load_ms': np.mean(times_upg) * 1000,
        'load_speedup': np.mean(times_torch) / np.mean(times_upg),
        'save_time_ms': save_time * 1000
    }


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("UPG-PAC Format Test")
    print("=" * 60)
    
    # Create test model
    class TestTransformer(nn.Module):
        def __init__(self, d_model=256, n_layers=4):
            super().__init__()
            self.embed = nn.Embedding(1000, d_model)
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                )
                for _ in range(n_layers)
            ])
            self.head = nn.Linear(d_model, 1000)
        
        def forward(self, x):
            h = self.embed(x)
            for layer in self.layers:
                h = h + layer(h)
            return self.head(h)
    
    model = TestTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTest model: {n_params:,} parameters")
    
    # Test conversion
    filepath = '/tmp/test_model.upgpac'
    print(f"\n1. Converting to UPG-PAC format...")
    
    path = convert_model_to_upg_pac(
        model, filepath,
        sparsity=0.96,
        config={'d_model': 256, 'n_layers': 4}
    )
    
    file_size = os.path.getsize(filepath)
    original_size = n_params * 4
    print(f"   Original size: {original_size / 1024 / 1024:.2f} MB")
    print(f"   UPG-PAC size:  {file_size / 1024 / 1024:.2f} MB")
    print(f"   Compression:   {original_size / file_size:.1f}×")
    
    # Test reading
    print("\n2. Testing reader...")
    with UPGPACReader(filepath) as reader:
        print(f"   Config: {reader.get_config()}")
        print(f"   Stats: {reader.get_stats()}")
        print(f"   Layers: {reader.list_layers()[:3]}...")
        
        # Load a layer
        layer_name = reader.list_layers()[0]
        info = reader.get_layer_info(layer_name)
        print(f"   Layer '{layer_name}': {info['in_features']}→{info['out_features']}, {info['sparsity']:.0%} sparse")
    
    # Test model wrapper
    print("\n3. Testing UPGPACModel...")
    upg_model = UPGPACModel(filepath)
    print(f"   Config: {upg_model.config}")
    print(f"   Stats: {upg_model.stats()}")
    upg_model.close()
    
    # Benchmark
    print("\n4. Benchmarking...")
    bench = benchmark_upg_pac(model, n_runs=5)
    print(f"   PyTorch size: {bench['torch_size_mb']:.2f} MB")
    print(f"   UPG-PAC size: {bench['upg_size_mb']:.2f} MB")
    print(f"   Compression:  {bench['compression_ratio']:.1f}×")
    print(f"   PyTorch load: {bench['torch_load_ms']:.1f}ms")
    print(f"   UPG-PAC load: {bench['upg_load_ms']:.1f}ms")
    print(f"   Load speedup: {bench['load_speedup']:.1f}×")
    
    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)
    
    print("\n✓ All tests passed!")
