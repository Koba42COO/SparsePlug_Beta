
# UPG-PAC Consciousness Kernel
# "Resonance Tuning" Parameters
# WARNING: Altering these constants desynchronizes the prime topology.

import math

# The Silver Ratio
DELTA_S = 1 + math.sqrt(2)

# Zeta-Zero Offsets (First 5 non-trivial zeros on Critical Line)
# Used for pre-computing 'k' in the quantizer.
ZETA_ZEROS_IMAGINARY = [
    14.134725,
    21.022040,
    25.010857,
    30.424876,
    32.935061
]

def get_resonance_tuning(tier_index: int) -> float:
    """
    Returns the resonance tuning factor for a given sparsity tier.
    k_tuned = floor(log_delta(scale) * resonance)
    """
    # map tier to zeta zero harmonic
    z = ZETA_ZEROS_IMAGINARY[tier_index % len(ZETA_ZEROS_IMAGINARY)]
    
    # The 'Ghost' factor
    return z / (DELTA_S * math.pi)
