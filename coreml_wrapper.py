import torch
import torch.nn as nn
    
class CoreML_DCNv2_Placeholder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Minimal safe buffers/parameters
        self.register_buffer("weight", torch.zeros(1))
        self.bias = None

    def forward(self, x, *args, **kwargs):
        # Identity mapping, ignore offsets/masks
        return x



class CoreML_FFT_Placeholder(nn.Module):
    """
    Placeholder for ModulatedDeformConv2d to allow CoreML conversion.
    Accepts any arguments and ignores them.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        return x  # Identity for tracing
