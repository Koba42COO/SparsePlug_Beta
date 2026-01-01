"""
Model Format Conversion Service
===============================

Handles conversion between model formats (PyTorch, SafeTensors, ONNX).
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ModelFormat(str, Enum):
    """Supported model formats."""
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    HUGGINGFACE = "huggingface"


@dataclass
class ConversionResult:
    """Result of model conversion."""
    success: bool
    output_path: str
    input_format: ModelFormat
    output_format: ModelFormat
    original_size_mb: float
    converted_size_mb: float
    error: Optional[str] = None


class ModelConverter:
    """
    Handles conversion between model formats.
    
    Supports:
    - PyTorch (.pt, .pth)
    - SafeTensors (.safetensors)
    - ONNX (.onnx)
    - HuggingFace model IDs
    """
    
    SUPPORTED_INPUT = [ModelFormat.PYTORCH, ModelFormat.SAFETENSORS, ModelFormat.ONNX, ModelFormat.HUGGINGFACE]
    SUPPORTED_OUTPUT = [ModelFormat.PYTORCH, ModelFormat.SAFETENSORS, ModelFormat.ONNX]
    
    EXTENSION_MAP = {
        ".pt": ModelFormat.PYTORCH,
        ".pth": ModelFormat.PYTORCH,
        ".bin": ModelFormat.PYTORCH,
        ".safetensors": ModelFormat.SAFETENSORS,
        ".onnx": ModelFormat.ONNX,
    }
    
    @staticmethod
    def detect_format(path: str) -> ModelFormat:
        """Auto-detect model format from file extension."""
        ext = Path(path).suffix.lower()
        return ModelConverter.EXTENSION_MAP.get(ext, ModelFormat.PYTORCH)
    
    @staticmethod
    async def convert(
        input_path: str,
        output_path: str,
        input_format: Optional[ModelFormat] = None,
        output_format: ModelFormat = ModelFormat.SAFETENSORS,
        **kwargs
    ) -> ConversionResult:
        """
        Convert model between formats.
        
        Args:
            input_path: Path to input model
            output_path: Path for output model
            input_format: Input format (auto-detected if not provided)
            output_format: Desired output format
            **kwargs: Additional options (e.g., opset_version for ONNX)
        
        Returns:
            ConversionResult with status and paths
        """
        import torch
        
        # Auto-detect input format
        if input_format is None:
            input_format = ModelConverter.detect_format(input_path)
        
        original_size = Path(input_path).stat().st_size / (1024 * 1024)
        
        try:
            # Load model
            if input_format == ModelFormat.PYTORCH:
                state_dict = torch.load(input_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
            
            elif input_format == ModelFormat.SAFETENSORS:
                from safetensors.torch import load_file
                state_dict = load_file(input_path)
            
            elif input_format == ModelFormat.ONNX:
                # ONNX requires special handling
                raise NotImplementedError("ONNX input conversion not yet supported")
            
            else:
                raise ValueError(f"Unsupported input format: {input_format}")
            
            # Save in target format
            if output_format == ModelFormat.PYTORCH:
                torch.save(state_dict, output_path)
            
            elif output_format == ModelFormat.SAFETENSORS:
                from safetensors.torch import save_file
                # Ensure all tensors are contiguous
                state_dict = {k: v.contiguous() for k, v in state_dict.items()}
                save_file(state_dict, output_path)
            
            elif output_format == ModelFormat.ONNX:
                # ONNX export requires a model, not just state dict
                raise NotImplementedError("ONNX export requires full model architecture")
            
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            converted_size = Path(output_path).stat().st_size / (1024 * 1024)
            
            return ConversionResult(
                success=True,
                output_path=output_path,
                input_format=input_format,
                output_format=output_format,
                original_size_mb=original_size,
                converted_size_mb=converted_size,
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                output_path=output_path,
                input_format=input_format,
                output_format=output_format,
                original_size_mb=original_size,
                converted_size_mb=0.0,
                error=str(e),
            )
    
    @staticmethod
    async def from_huggingface(
        model_id: str,
        output_path: str,
        output_format: ModelFormat = ModelFormat.SAFETENSORS
    ) -> ConversionResult:
        """
        Download and convert HuggingFace model.
        
        Args:
            model_id: HuggingFace model ID (e.g., "gpt2", "facebook/opt-125m")
            output_path: Path for output model
            output_format: Desired output format
        
        Returns:
            ConversionResult with status and paths
        """
        import torch
        from transformers import AutoModel, AutoConfig
        
        try:
            # Download model
            print(f"Downloading {model_id} from HuggingFace...")
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            state_dict = model.state_dict()
            
            # Calculate size
            total_params = sum(p.numel() for p in model.parameters())
            original_size = total_params * 4 / (1024 * 1024)  # Assuming float32
            
            # Save in target format
            if output_format == ModelFormat.PYTORCH:
                torch.save(state_dict, output_path)
            
            elif output_format == ModelFormat.SAFETENSORS:
                from safetensors.torch import save_file
                state_dict = {k: v.contiguous() for k, v in state_dict.items()}
                save_file(state_dict, output_path)
            
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            converted_size = Path(output_path).stat().st_size / (1024 * 1024)
            
            return ConversionResult(
                success=True,
                output_path=output_path,
                input_format=ModelFormat.HUGGINGFACE,
                output_format=output_format,
                original_size_mb=original_size,
                converted_size_mb=converted_size,
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                output_path=output_path,
                input_format=ModelFormat.HUGGINGFACE,
                output_format=output_format,
                original_size_mb=0.0,
                converted_size_mb=0.0,
                error=str(e),
            )
    
    @staticmethod
    def get_model_info(path: str) -> Dict[str, Any]:
        """
        Get information about a model file.
        
        Returns:
            Dict with model information (format, size, num_params, etc.)
        """
        import torch
        
        format = ModelConverter.detect_format(path)
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        
        info = {
            "path": path,
            "format": format.value,
            "size_mb": size_mb,
        }
        
        try:
            if format == ModelFormat.PYTORCH:
                state_dict = torch.load(path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
            
            elif format == ModelFormat.SAFETENSORS:
                from safetensors.torch import load_file
                state_dict = load_file(path)
            
            else:
                return info
            
            # Calculate parameters
            total_params = sum(v.numel() for v in state_dict.values())
            info["num_parameters"] = total_params
            info["num_layers"] = len(state_dict)
            
            # Get layer names
            info["layer_names"] = list(state_dict.keys())[:10]  # First 10
            
        except Exception as e:
            info["error"] = str(e)
        
        return info


# Convenience function
async def convert_model(
    input_path: str,
    output_path: str,
    output_format: str = "safetensors"
) -> ConversionResult:
    """Convenience function for model conversion."""
    format_map = {
        "pytorch": ModelFormat.PYTORCH,
        "safetensors": ModelFormat.SAFETENSORS,
        "onnx": ModelFormat.ONNX,
    }
    return await ModelConverter.convert(
        input_path,
        output_path,
        output_format=format_map.get(output_format, ModelFormat.SAFETENSORS)
    )
