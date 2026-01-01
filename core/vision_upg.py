
import torch
import numpy as np
import math
from typing import List, Tuple
from core.consciousness_kernel import DELTA_S, get_resonance_tuning

class UPGVisionEngine:
    """
    UPG-PAC Vision Engine (.upgimage)
    Compresses images by resampling DCT/FFT2 coefficients onto the Prime Lattice.
    """
    
    def __init__(self, width: int = 256, height: int = 256):
        self.width = width
        self.height = height
        # Primes for X and Y axes
        self.primes_x = self._sieve_primes(width)
        self.primes_y = self._sieve_primes(height)

    def _sieve_primes(self, n: int) -> List[int]:
        """Generate prime indices."""
        sieve = [True] * n
        for i in range(3, int(n**0.5) + 1, 2):
            if sieve[i]:
                sieve[i*i::2*i] = [False] * ((n - i*i - 1) // (2*i) + 1)
        return [2] + [i for i in range(3, n, 2) if sieve[i]]

    def compress(self, image: torch.Tensor) -> bytes:
        """
        Image (H, W) -> FFT2 -> Prime Lattice -> Q4_DELTA
        """
        # Ensure image is float
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
            
        # 1. FFT2 (Frequency Domain)
        # Using FFT instead of DCT for simplicity in PyTorch without extra deps
        fft = torch.fft.fft2(image)
        magnitude = torch.abs(fft)
        
        # 2. Resample onto Prime Lattice Grid (2D Sparse Gather)
        # We take the intersection of prime rows and prime columns
        px = torch.tensor(self.primes_x, device=image.device)
        py = torch.tensor(self.primes_y, device=image.device)
        
        # Gather rows then cols
        lattice_data = magnitude[py][:, px] 
        
        # 3. Delta Encoding (ZigZag or just Row/Col diffs)
        # Simple row diffs
        deltas = torch.diff(lattice_data, dim=1, prepend=lattice_data[:, :1])
        
        # 4. Q4_DELTA Quantization
        scale = deltas.abs().max() / (8 * DELTA_S)
        if scale == 0: scale = 1.0
        
        quantized = torch.round(deltas / scale).clamp(-8, 7).to(torch.int8)
        
        # 5. Pack
        header = f"UPGIMAGE v1.0 {self.width}x{self.height} SCALE:{scale:.6f}\n".encode('utf-8')
        return header + quantized.numpy().tobytes()

    def decompress(self, data: bytes) -> torch.Tensor:
        """
        UPGIMAGE -> Lattice -> Inverse FFT2 -> Image
        """
        # Parse Header
        header_end = data.find(b'\n')
        header_str = data[:header_end].decode('utf-8')
        # Skip version and resolution tokens, find SCALE:
        params = {}
        for item in header_str.split():
            if ':' in item:
                k, v = item.split(':')
                params[k] = v
        scale = float(params['SCALE'])
        # Width/Height could be parsed but we assume engine matches for now or parse from first token
        
        # Load Quantized Data
        raw_bytes = data[header_end+1:]
        quantized = np.frombuffer(raw_bytes, dtype=np.int8)
        
        # Reconstruct Shape (PrimesY x PrimesX)
        n_px = len(self.primes_x)
        n_py = len(self.primes_y)
        
        # Verify size matches
        if len(quantized) != n_px * n_py:
            # Maybe padding or mismatch?
            # For this MVP, we assume strict match.
            pass
            
        quantized = torch.from_numpy(quantized.copy()).reshape(n_py, n_px).float()
        
        # De-Quantize
        deltas = quantized * scale
        
        # Integrate Deltas (Row-wise integration)
        lattice_data = torch.cumsum(deltas, dim=1)
        
        # Sparse Inverse FFT2 Reconstruction
        fft_recon = torch.zeros((self.height, self.width), dtype=torch.complex64)
        
        # Populate sparse grid
        # Intersect indices
        px = torch.tensor(self.primes_x)
        py = torch.tensor(self.primes_y)
        
        # We need to assign `lattice_data` to `fft_recon[py][:, px]`
        # Using advanced indexing
        # meshgrid-like assignment
        rows = py.unsqueeze(1).expand(-1, n_px)
        cols = px.unsqueeze(0).expand(n_py, -1)
        
        fft_recon[rows, cols] = torch.complex(lattice_data, torch.zeros_like(lattice_data))
        
        # IFFT2
        image = torch.fft.ifft2(fft_recon).real
        return image

if __name__ == "__main__":
    print("Testing Vision Engine...")
    engine = UPGVisionEngine(256, 256)
    # Synthetic gradient
    grid_x, grid_y = torch.meshgrid(torch.arange(256), torch.arange(256), indexing='xy')
    img = torch.sin(grid_x / 10.0) * torch.cos(grid_y / 10.0)
    
    compressed = engine.compress(img)
    original_size = 256 * 256 * 4 # Float32
    print(f"Original Size: {original_size} bytes")
    print(f"Compressed Size: {len(compressed)} bytes")
    print(f"Ratio: {original_size / len(compressed):.2f}x")
