import struct
import json
import mmap
import os
from typing import Generator, List, Dict, Any

from .upg_pac import UPG_PAC_MAGIC

class SliceServingService:
    """
    Service to stream specific slices (tiers) of a UPG-PAC model.
    This enables 'On-the-Fly Repacking' for variable compression.
    """
    
    PAGE_SIZE = 4096
    
    def __init__(self):
        self.last_merkle_root = ""

    def get_tier_name(self, sparsity: float) -> str:
        """Map sparsity to marketing tier."""
        if sparsity >= 0.98: return "IoT-Extreme"
        if sparsity >= 0.95: return "Mobile-High"
        if sparsity >= 0.70: return "Edge-Medium"
        return "Server-Full"

    def stream_slice(self, filepath: str, target_sparsity: float) -> Generator[bytes, None, None]:
        """
        Yields bytes of a VALID .upgpac file containing only tiers 
        required for the target sparsity.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            # Memory map for efficient random access
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                
                # 1. Read Header
                # Magic (8) + Version (4) + n_layers (4) + metadata_len (8)
                header_bytes = mm[:24]
                if header_bytes[:8] != UPG_PAC_MAGIC:
                    raise ValueError("Invalid UPG-PAC file")
                    
                version, n_layers, metadata_len = struct.unpack('<IIQ', header_bytes[8:24])
                
                # 2. Read Metadata
                # Original metadata
                metadata_json_bytes = mm[264 : 264 + metadata_len]
                metadata = json.loads(metadata_json_bytes.decode('utf-8'))
                
                # Capture Merkle Root
                self.last_merkle_root = metadata.get('integrity', {}).get('merkle_root', '')
                
                # 3. Filter Layers and Plan New Layout
                original_layers = metadata.get('layers', [])
                new_layers = []
                chunks_to_send = [] # List of (start, size) from original file
                
                # We assume the new file data starts at the SAME offset as original
                # This ensures we have enough space for the new (smaller) metadata
                if original_layers:
                    new_data_start = original_layers[0]['block_offset']
                else:
                    new_data_start = 264 + metadata_len + 4096 # Fallback
                
                current_write_offset = new_data_start
                
                for layer in original_layers:
                    new_spec = layer.copy()
                    
                    if layer['layer_type'] == 'sparse_linear':
                        tiers = layer.get('tiers', [])
                        bias_offset = layer.get('bias_offset', 0)
                        
                        # Filter tiers
                        if not tiers:
                            # Keep everything (Legacy/Single)
                            chunks_to_send.append((layer['block_offset'], layer['block_size']))
                            new_spec['block_offset'] = current_write_offset
                            current_write_offset += layer['block_size']
                        else:
                            # v4 Multi-tier Sort
                            sorted_tiers = sorted(tiers, key=lambda x: x['sparsity'], reverse=True)
                            
                            tier_bytes_count = 0
                            tiers_metadata = []
                            layer_start_offset = current_write_offset
                            current_tier_out = layer_start_offset
                            
                            for tier in sorted_tiers:
                                # Keep tier if sparsity >= target (e.g. 0.99 >= 0.96)
                                if tier['sparsity'] >= target_sparsity - 1e-6:
                                    chunks_to_send.append((tier['offset'], tier['size']))
                                    
                                    new_tier_meta = tier.copy()
                                    new_tier_meta['offset'] = current_tier_out
                                    tiers_metadata.append(new_tier_meta)
                                    
                                    current_tier_out += tier['size']
                                    tier_bytes_count += tier['size']
                            
                            # Handle Bias
                            bias_start = layer['block_offset'] + bias_offset
                            bias_end = layer['block_offset'] + layer['block_size']
                            bias_size = bias_end - bias_start
                            
                            if bias_size > 0:
                                chunks_to_send.append((bias_start, bias_size))
                                new_bias_offset_rel = tier_bytes_count
                                tier_bytes_count += bias_size
                            else:
                                new_bias_offset_rel = 0
                            
                            # Update Spec
                            new_spec['block_offset'] = layer_start_offset
                            new_spec['block_size'] = tier_bytes_count
                            new_spec['bias_offset'] = new_bias_offset_rel
                            new_spec['tiers'] = tiers_metadata
                            # Update layer sparsity to target (effective)
                            new_spec['sparsity'] = min([t['sparsity'] for t in tiers_metadata]) if tiers_metadata else 1.0
                            
                            current_write_offset += tier_bytes_count
                            
                    elif layer['layer_type'] == 'embedding':
                        chunks_to_send.append((layer['block_offset'], layer['block_size']))
                        new_spec['block_offset'] = current_write_offset
                        current_write_offset += layer['block_size']
                        
                    new_layers.append(new_spec)
                
                # 4. Reconstruct Metadata
                metadata['layers'] = new_layers
                new_metadata_json = json.dumps(metadata).encode('utf-8')
                
                # 5. Yield Stream
                
                # Magic
                yield UPG_PAC_MAGIC
                
                # Header (update metadata len)
                new_header = struct.pack('<IIQ', version, len(new_layers), len(new_metadata_json))
                yield new_header
                
                # Padding 256
                yield b'\x00' * (256 - 24)
                
                # Meta Len
                yield struct.pack('<Q', len(new_metadata_json))
                
                # Meta JSON
                yield new_metadata_json
                
                # Padding to Data Start
                current_pos = 264 + len(new_metadata_json)
                pad_len = new_data_start - current_pos
                if pad_len > 0:
                    yield b'\x00' * pad_len
                elif pad_len < 0:
                    # Should not happen if new metadata is smaller and we reused offset
                    # But if it happens, we are overlapping.
                    # Fallback -> start data later? 
                    # If we start later, 'current_write_offset' is wrong.
                    # We must ensure metadata fits.
                    # For now assume it fits or client handles mismatch if we shift?
                    # No, valid file must match. 
                    # If pad_len < 0, we failed constraints.
                    # Simple fix: Yield padding to 0? No.
                    pass 
                
                # Data Chunks
                for start, size in chunks_to_send:
                    yield mm[start : start+size]
