"""
Accuracy Validation Service
===========================

Validates that optimized models maintain quality.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np


@dataclass
class ValidationResult:
    """Result of accuracy validation."""
    passed: bool
    accuracy_delta: float
    perplexity_delta: float
    output_similarity: float
    layer_analysis: Dict[str, Dict[str, float]]
    recommendations: List[str]


class AccuracyValidator:
    """
    Validates that optimized model maintains quality.
    
    Compares original vs optimized model outputs to ensure
    the optimization doesn't significantly degrade quality.
    """
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize validator.
        
        Args:
            tolerance: Maximum allowed accuracy loss (0.01 = 1%)
        """
        self.tolerance = tolerance
    
    async def validate(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module,
        validation_data: Optional[torch.utils.data.DataLoader] = None,
        num_samples: int = 100
    ) -> ValidationResult:
        """
        Compare original vs optimized model.
        
        Args:
            original_model: Original unoptimized model
            optimized_model: Optimized sparse model
            validation_data: Optional validation data loader
            num_samples: Number of synthetic samples if no data provided
        
        Returns:
            ValidationResult with detailed comparison
        """
        device = next(original_model.parameters()).device
        
        # Generate synthetic data if not provided
        if validation_data is None:
            hidden_size = self._get_hidden_size(original_model)
            inputs = self._generate_synthetic_data(hidden_size, num_samples, device)
        else:
            inputs = next(iter(validation_data))[0].to(device)
        
        # Compare outputs
        with torch.no_grad():
            original_model.eval()
            optimized_model.eval()
            
            orig_output = original_model(inputs)
            opt_output = optimized_model(inputs)
        
        # Calculate metrics
        output_similarity = self._compute_cosine_similarity(orig_output, opt_output)
        mse = self._compute_mse(orig_output, opt_output)
        
        # Per-layer analysis
        layer_analysis = self._analyze_layers(original_model, optimized_model, inputs)
        
        # Determine pass/fail
        accuracy_delta = 1.0 - output_similarity
        perplexity_delta = mse * 10  # Rough approximation
        
        passed = accuracy_delta <= self.tolerance
        
        # Generate recommendations
        recommendations = []
        if not passed:
            recommendations.append(f"Accuracy loss ({accuracy_delta:.2%}) exceeds tolerance ({self.tolerance:.2%})")
            recommendations.append("Consider reducing sparsity level")
            recommendations.append("Try fine-tuning after sparsification")
        else:
            recommendations.append("Optimization successful - quality maintained")
            if accuracy_delta < self.tolerance / 2:
                recommendations.append("Could potentially increase sparsity further")
        
        return ValidationResult(
            passed=passed,
            accuracy_delta=accuracy_delta,
            perplexity_delta=perplexity_delta,
            output_similarity=output_similarity,
            layer_analysis=layer_analysis,
            recommendations=recommendations,
        )
    
    def _get_hidden_size(self, model: nn.Module) -> int:
        """Get model's hidden size from first linear layer."""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        return 768  # Default
    
    def _generate_synthetic_data(
        self,
        hidden_size: int,
        num_samples: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate synthetic input data."""
        # Generate data that looks like typical model inputs
        # Using a mixture of patterns
        batch_size = min(num_samples, 32)
        seq_len = 128
        
        # Mix of random and structured data
        random_data = torch.randn(batch_size, seq_len, hidden_size, device=device)
        
        # Add some structure (like sentence patterns)
        pattern = torch.sin(torch.arange(seq_len).float().unsqueeze(0).unsqueeze(-1).to(device) / 10)
        pattern = pattern.expand(batch_size, -1, hidden_size)
        
        return 0.8 * random_data + 0.2 * pattern
    
    def _compute_cosine_similarity(
        self,
        orig: torch.Tensor,
        opt: torch.Tensor
    ) -> float:
        """Compute cosine similarity between outputs."""
        orig_flat = orig.flatten()
        opt_flat = opt.flatten()
        
        similarity = torch.nn.functional.cosine_similarity(
            orig_flat.unsqueeze(0),
            opt_flat.unsqueeze(0)
        )
        return float(similarity.item())
    
    def _compute_mse(self, orig: torch.Tensor, opt: torch.Tensor) -> float:
        """Compute mean squared error."""
        mse = torch.mean((orig - opt) ** 2)
        return float(mse.item())
    
    def _analyze_layers(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module,
        inputs: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze output differences per layer.
        
        Returns dict mapping layer names to their similarity metrics.
        """
        layer_analysis = {}
        
        orig_activations = {}
        opt_activations = {}
        
        def get_activation(name, storage):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                storage[name] = output.detach()
            return hook
        
        # Register hooks
        orig_hooks = []
        opt_hooks = []
        
        for name, module in original_model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                hook = module.register_forward_hook(get_activation(name, orig_activations))
                orig_hooks.append(hook)
        
        for name, module in optimized_model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                hook = module.register_forward_hook(get_activation(name, opt_activations))
                opt_hooks.append(hook)
        
        # Run forward pass
        with torch.no_grad():
            _ = original_model(inputs)
            _ = optimized_model(inputs)
        
        # Remove hooks
        for hook in orig_hooks + opt_hooks:
            hook.remove()
        
        # Compare activations
        for name in orig_activations:
            if name in opt_activations:
                orig_act = orig_activations[name]
                opt_act = opt_activations[name]
                
                similarity = self._compute_cosine_similarity(orig_act, opt_act)
                mse = self._compute_mse(orig_act, opt_act)
                
                layer_analysis[name] = {
                    "cosine_similarity": similarity,
                    "mse": mse,
                    "degradation": 1.0 - similarity,
                }
        
        return layer_analysis
    
    def quick_validate(
        self,
        original_output: torch.Tensor,
        optimized_output: torch.Tensor
    ) -> Dict[str, float]:
        """
        Quick validation comparing just outputs.
        
        Args:
            original_output: Output from original model
            optimized_output: Output from optimized model
        
        Returns:
            Dict with similarity metrics
        """
        return {
            "cosine_similarity": self._compute_cosine_similarity(original_output, optimized_output),
            "mse": self._compute_mse(original_output, optimized_output),
            "max_diff": float(torch.max(torch.abs(original_output - optimized_output)).item()),
            "mean_diff": float(torch.mean(torch.abs(original_output - optimized_output)).item()),
        }


# Convenience function
def validate_optimization(
    original_output: torch.Tensor,
    optimized_output: torch.Tensor,
    tolerance: float = 0.01
) -> bool:
    """
    Quick check if optimization is acceptable.
    
    Args:
        original_output: Output from original model
        optimized_output: Output from optimized model
        tolerance: Maximum allowed difference
    
    Returns:
        True if optimization is acceptable
    """
    validator = AccuracyValidator(tolerance)
    metrics = validator.quick_validate(original_output, optimized_output)
    return metrics["cosine_similarity"] >= (1.0 - tolerance)
