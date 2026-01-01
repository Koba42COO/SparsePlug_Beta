
import torch
import numpy as np
import math
from typing import List, Tuple
from core.consciousness_kernel import DELTA_S, get_resonance_tuning

class UPGAudioEngine:
    """
    UPG-PAC Audio Engine (.upgtrack)
    Compresses audio by resampling spectral peaks onto the Prime Lattice.
    """
    
    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self.primes = self._sieve_primes(n_fft // 2)

    def _sieve_primes(self, n: int) -> List[int]:
        """Generate prime indices for the lattice."""
        sieve = [True] * n
        for i in range(3, int(n**0.5) + 1, 2):
            if sieve[i]:
                sieve[i*i::2*i] = [False] * ((n - i*i - 1) // (2*i) + 1)
        return [2] + [i for i in range(3, n, 2) if sieve[i]]

    def compress(self, waveform: torch.Tensor) -> bytes:
        """
        Waveform -> STFT -> Prime Lattice -> Q4_DELTA
        """
        # 1. STFT
        stft = torch.stft(waveform, self.n_fft, hop_length=self.hop_length, return_complex=True)
        magnitude = torch.abs(stft)
        
        # 2. Resample onto Prime Lattice
        # We only keep rows corresponding to prime indices
        prime_indices = torch.tensor(self.primes, device=waveform.device)
        # Handle index out of bounds if n_fft/2 < max prime? 
        # Primes are generated up to n_fft//2, so it's safe.
        
        lattice_data = magnitude[prime_indices, :]
        
        # 3. Delta Encoding across Time
        # Calculate differences between adjacent timeframes to minimize entropy
        deltas = torch.diff(lattice_data, dim=1, prepend=lattice_data[:, :1])
        
        # 4. Q4_DELTA Quantization (Silver Ratio)
        # Using a simplified 1-tier approach for this demo layer.
        # Scale factor typically determined by dynamic range, here we use fixed or max.
        scale = deltas.abs().max() / (8 * DELTA_S) # 4-bit range approx [-8, 7]
        
        if scale == 0: scale = 1.0
        
        quantized = torch.round(deltas / scale).clamp(-8, 7).to(torch.int8)
        
        # 5. Pack Header and Data
        header = f"UPGTRACK v1.0 SR:{self.sample_rate} LEN:{waveform.numel()} SCALE:{scale:.6f}\n".encode('utf-8')
        return header + quantized.numpy().tobytes()

    def decompress(self, data: bytes) -> torch.Tensor:
        """
        UPGTRACK -> Lattice -> Inverse STFT -> Waveform
        """
        # Parse Header
        header_end = data.find(b'\n')
        header_str = data[:header_end].decode('utf-8')
        params = dict(item.split(':') for item in header_str.split()[2:])
        sr = int(params['SR'])
        original_len = int(params['LEN'])
        scale = float(params['SCALE'])
        
        # Load Quantized Data
        raw_bytes = data[header_end+1:]
        quantized = np.frombuffer(raw_bytes, dtype=np.int8)
        
        # Reconstruct Shape (Primes x Time)
        n_primes = len(self.primes)
        n_frames = len(quantized) // n_primes
        quantized = torch.from_numpy(quantized).reshape(n_primes, n_frames).float()
        
        # De-Quantize
        deltas = quantized * scale
        
        # Integrate Deltas to recover Magnitude
        lattice_data = torch.cumsum(deltas, dim=1)
        
        # Sparse Inverse STFT Reconstruction
        # We start with zeros and fill prime rows
        stft_recon = torch.zeros((self.n_fft // 2 + 1, n_frames), dtype=torch.complex64)
        
        # We model phase as zero (magnitude-only reconstruction, "Griffin-Lim" style could be added)
        # Or simple "Robot Voice" reconstruction for MVP.
        stft_recon[self.primes, :] = torch.complex(lattice_data, torch.zeros_like(lattice_data)) 
        
        # ISTFT
        waveform = torch.istft(stft_recon, self.n_fft, hop_length=self.hop_length, length=original_len)
        return waveform

if __name__ == "__main__":
    # Test
    print("Testing Audio Engine...")
    engine = UPGAudioEngine()
    # 1 second sine sweep
    t = torch.linspace(0, 1, 44100)
    wave = torch.sin(2 * math.pi * 440 * t) 
    
    compressed = engine.compress(wave)
    print(f"Original Size: {wave.numel() * 4} bytes")
    print(f"Compressed Size: {len(compressed)} bytes")
    print(f"Ratio: {wave.numel() * 4 / len(compressed):.2f}x")
