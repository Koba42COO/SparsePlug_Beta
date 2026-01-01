import struct
import json
import mmap
import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

UPG_DATA_MAGIC = b'UPG_DATA'
UPG_DATA_VERSION = 1

class UPGDatasetWriter:
    """
    Writes datasets in the UPG-Data format (.upgdata).
    
    Format:
    [MAGIC:8][VERSION:4][COUNT:8][META_LEN:8][PADDING:228] -> Header (256 bytes)
    [METADATA_JSON]
    [DATA_CATACOMBS...] -> Packed tensors
    """
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.samples = []
        self.metadata = {'samples': []}
        # Temporary file for data to append later? 
        # Or keep in memory if small? 
        # For large datasets, we should write data sequentially and keep index in memory.
        
        self.f = open(self.filepath, 'wb')
        # Placeholder header
        self.f.write(b'\x00' * 256)
        self.current_offset = 256
        self.count = 0
        
    def add(self, tensor: torch.Tensor, meta: Dict = None):
        """Add a tensor sample to the dataset."""
        # Convert to numpy
        if isinstance(tensor, torch.Tensor):
            data = tensor.detach().cpu().numpy()
        else:
            data = np.array(tensor)
            
        # Serialize
        data_bytes = data.tobytes()
        
        # Write
        start_offset = self.f.tell()
        self.f.write(data_bytes)
        size = len(data_bytes)
        
        # Record
        sample_info = {
            'id': self.count,
            'offset': start_offset,
            'size': size,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'meta': meta or {}
        }
        self.samples.append(sample_info)
        self.count += 1
        
    def close(self):
        """Finalize the file."""
        if not self.f: return
        
        # Write Metadata at the END? 
        # If we write at end, we need pointer in header.
        # Or we can just standard append.
        
        # Let's write metadata at the end (Footer style) for append-ability?
        # Header has fixed size, so we can't put variable metadata there easily if it grows.
        # But for reading, front is better.
        # Let's write metadata after data, and update header to point to it?
        # Standard: 
        # [Header -> Ptr to Meta] ... [Data] ... [Meta]
        
        meta_start = self.f.tell()
        meta_json = json.dumps({'samples': self.samples})
        meta_bytes = meta_json.encode('utf-8')
        self.f.write(meta_bytes)
        meta_len = len(meta_bytes)
        
        # Go back to header
        self.f.seek(0)
        # Magic (8) + Version (4) + Count (8) + MetaStart (8) + MetaLen (8)
        header = struct.pack('<8sIQQQ', UPG_DATA_MAGIC, UPG_DATA_VERSION, self.count, meta_start, meta_len)
        self.f.write(header)
        
        self.f.close()
        self.f = None
        print(f"Dataset written to {self.filepath}: {self.count} samples.")

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()


class UPGDatasetReader(torch.utils.data.Dataset):
    """
    PyTorch Dataset for reading .upgdata files.
    Directly maps to memory for zero-copy access.
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found")
            
        self.f = open(filepath, 'rb')
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Read Header
        header = self.mm[:256]
        magic, version, count, meta_start, meta_len = struct.unpack('<8sIQQQ', header[:36])
        
        if magic != UPG_DATA_MAGIC:
            raise ValueError("Invalid UPG-Data file")
            
        self.count = count
        
        # Read Metadata
        meta_bytes = self.mm[meta_start : meta_start + meta_len]
        self.metadata = json.loads(meta_bytes.decode('utf-8'))
        self.samples_info = self.metadata['samples']
        
    def __len__(self):
        return self.count
        
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.count:
            raise IndexError("Index out of bounds")
            
        info = self.samples_info[idx] # Bug fix: samples_info vs samples_ (json key is 'samples')
        # Ah wait, self.samples_info = self.metadata['samples']
        
        offset = info['offset']
        size = info['size']
        shape = info['shape']
        dtype_str = info['dtype']
        
        # Read raw
        raw_bytes = self.mm[offset : offset + size]
        
        # Reconstruct numpy
        # safer to use np.frombuffer
        dtype = np.dtype(dtype_str)
        arr = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
        
        # Convert to torch (copy required usually for collation, but we want tensor)
        return torch.from_numpy(arr.copy()) # Copy to ensure writable if needed
        
    def close(self):
        if self.mm: self.mm.close()
        if self.f: self.f.close()
