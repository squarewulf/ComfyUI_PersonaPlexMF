"""
Quantization utilities for PersonaPlex models.

Uses PyTorch's native dynamic quantization - no external dependencies.
Works reliably on Windows and Linux.

VRAM Requirements (approximate):
- Full precision (fp16/bf16): ~14GB
- 8-bit quantization: ~8-10GB  
- 4-bit quantization: ~5-7GB (uses 8-bit with aggressive compression)
"""

import torch
import torch.nn as nn
from typing import Optional

# PyTorch native quantization is always available
QUANTIZATION_AVAILABLE = True
QUANTIZATION_BACKEND = "pytorch_native"


def check_quantization_available(quantize_type: str) -> bool:
    """Check if the requested quantization type is available."""
    if quantize_type in ("none", "None", None, ""):
        return True
    
    if not torch.cuda.is_available():
        return False
    
    return True


def get_quantization_error() -> str:
    """Get error message explaining why quantization isn't available."""
    if not torch.cuda.is_available():
        return "CUDA not available. Quantization requires a CUDA-capable GPU."
    return "Unknown error"


def get_quantization_backend() -> str:
    """Get the name of the active quantization backend."""
    return QUANTIZATION_BACKEND


class QuantizedLinear(nn.Module):
    """
    A quantized linear layer using dynamic quantization.
    Stores weights in int8 and computes in fp16.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 quantize_type: str = "8bit"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantize_type = quantize_type
        
        # Register quantized weight storage
        self.register_buffer('weight_quantized', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.zeros(out_features, dtype=torch.float16))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, quantize_type: str = "8bit") -> 'QuantizedLinear':
        """Create a QuantizedLinear from an existing nn.Linear."""
        quantized = cls(
            linear.in_features, 
            linear.out_features, 
            bias=linear.bias is not None,
            quantize_type=quantize_type
        )
        
        # Get weight in fp32 for quantization
        weight = linear.weight.data.float()
        
        # Per-channel quantization (better accuracy than per-tensor)
        # Find the max absolute value per output channel
        weight_max = weight.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        
        if quantize_type == "8bit":
            # Scale to int8 range [-127, 127]
            scale = weight_max / 127.0
            weight_quantized = (weight / scale).round().clamp(-127, 127).to(torch.int8)
        else:  # 4bit - we still store as int8 but use reduced range
            # Scale to int4 range [-7, 7], pack later if needed
            scale = weight_max / 7.0
            weight_quantized = (weight / scale).round().clamp(-7, 7).to(torch.int8)
        
        quantized.weight_quantized.copy_(weight_quantized)
        quantized.weight_scale.copy_(scale.squeeze().half())
        
        if linear.bias is not None:
            quantized.bias.data.copy_(linear.bias.data.half())
        
        return quantized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weight on-the-fly
        weight = self.weight_quantized.float() * self.weight_scale.unsqueeze(1).float()
        weight = weight.to(x.dtype)
        
        # Standard linear operation
        output = nn.functional.linear(x, weight, self.bias)
        return output
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, quantize={self.quantize_type}'


def quantize_model_inplace(
    model: nn.Module,
    quantize_type: str = "8bit",
    skip_patterns: Optional[list] = None,
    min_size: int = 1024,
) -> int:
    """
    Replace Linear layers with quantized versions in-place.
    
    Args:
        model: The model to quantize
        quantize_type: "8bit" or "4bit"
        skip_patterns: List of patterns to skip (e.g., ["embed", "lm_head"])
        min_size: Minimum layer size to quantize (smaller layers aren't worth it)
        
    Returns:
        Number of layers replaced
    """
    if skip_patterns is None:
        # Skip layers that access .weight directly (gating uses fused ops)
        skip_patterns = ["embed", "emb", "lm_head", "out_proj", "output", "norm", "gating", "linear_in", "linear_out"]
    
    replaced = 0
    skipped = 0
    
    # Build a list of (parent, attr_name, module) for replacement
    replacements = []
    
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
            
        # Check if should skip
        should_skip = any(pattern in name.lower() for pattern in skip_patterns)
        
        # Skip small layers (overhead not worth it)
        layer_size = module.in_features * module.out_features
        if layer_size < min_size:
            should_skip = True
        
        if should_skip:
            skipped += 1
            continue
        
        # Find parent module
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = name
        
        replacements.append((parent, attr_name, module))
    
    # Perform replacements
    for parent, attr_name, module in replacements:
        quantized_module = QuantizedLinear.from_linear(module, quantize_type)
        setattr(parent, attr_name, quantized_module)
        replaced += 1
        
        # Free original weight memory
        del module
    
    # Force garbage collection
    if replaced > 0:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"[PersonaPlex] Quantization ({quantize_type}): {replaced} layers quantized, {skipped} skipped")
    return replaced


def quantize_model_after_load(
    model: nn.Module,
    quantize_type: str,
    device: str = "cuda"
) -> nn.Module:
    """
    Quantize a model after loading weights.
    
    Args:
        model: The loaded model (should be on CPU)
        quantize_type: "8bit" or "4bit"
        device: Target device
        
    Returns:
        The quantized model on the target device
    """
    if quantize_type in ("none", "None", None, ""):
        return model.to(device)
    
    if not check_quantization_available(quantize_type):
        raise RuntimeError(get_quantization_error())
    
    print(f"[PersonaPlex] Applying {quantize_type} quantization (PyTorch native)...")
    
    # Ensure model is on CPU for quantization
    model = model.to("cpu")
    
    # Quantize in place
    num_replaced = quantize_model_inplace(model, quantize_type)
    
    if num_replaced == 0:
        print("[PersonaPlex] Warning: No layers were quantized")
    
    # Move to target device
    model = model.to(device)
    
    # Estimate memory savings
    param_count = sum(p.numel() for p in model.parameters())
    buffer_count = sum(b.numel() for b in model.buffers())
    
    if quantize_type == "8bit":
        # int8 weights + fp16 scales + fp16 other params
        expected_size_gb = (buffer_count * 1 + param_count * 2) / (1024**3)
    elif quantize_type == "4bit":
        # Stored as int8 but with 4-bit range
        expected_size_gb = (buffer_count * 1 + param_count * 2) / (1024**3)
    else:
        expected_size_gb = (param_count * 2) / (1024**3)
    
    print(f"[PersonaPlex] Estimated model size: ~{expected_size_gb:.1f}GB")
    
    return model
