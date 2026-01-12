import torch
import torch.nn as nn

class CoreML_DCNv2_Placeholder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, deform_groups, has_bias):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if has_bias: self.bias = nn.Parameter(torch.Tensor(out_channels))
        else: self.register_parameter('bias', None)
        self.stride, self.padding, self.dilation = stride, padding, dilation
        self.groups, self.deform_groups = groups, deform_groups

    def forward(self, x, offset, mask):
        return x # Identity for tracing

class CoreML_FFT_Placeholder(nn.Module):
    """
    Bypasses the torch.fft.rfft2 calls which CoreML cannot convert.
    """
    def __init__(self, out_channels):
        super().__init__()
        # This matches the internal conv_angle layers in RFFTAlignment
        self.conv_angle = nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x, feat_current):
        return x # Identity for tracing