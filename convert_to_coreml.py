import torch
import coremltools as ct
from coreml_wrapper import CoreML_DCNv2_Placeholder, CoreML_FFT_Placeholder

# Patch MMCV imports first
import sys
import types
sys.modules['mmcv.ops'] = types.SimpleNamespace(
    ModulatedDeformConv2d=CoreML_DCNv2_Placeholder,
    modulated_deform_conv2d=lambda *args, **kwargs: None
)

# Now safe to import the model
from model_2D.models.backbones.sr_backbones.DPA_TISR import DPATISR

# 1. Initialize
model = DPATISR(mid_channels=64, factor=2, bayesian=True)

# 2. Patch DCN and FFT Alignment
def patch_model(module):
    for name, child in module.named_children():
        # Patch DCNv2
        if "ModulatedDeformConv2d" in str(type(child)):
            setattr(module, name, CoreML_DCNv2_Placeholder(
                child.in_channels, child.out_channels, child.kernel_size[0], 
                child.stride[0], child.padding[0], child.dilation[0], 
                child.groups, child.deform_groups, child.bias is not None))
        # Patch FFT Alignment
        elif "RFFTAlignment" in str(type(child)):
            setattr(module, name, CoreML_FFT_Placeholder(child.out_channels))
        else:
            patch_model(child)

print("Patching DCN and FFT modules...")
patch_model(model)

# 3. Load Weights
checkpoint = torch.load('checkpt/MT_checkpt.pt', map_location='cpu')
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
model.load_state_dict(state_dict, strict=False)
model.eval()

# 4. Script the model instead of tracing
print("Scripting the model...")
s_model = torch.jit.script(model)

# 5. Conversion
print("Converting to CoreML...")
h, w = 256, 256  # fixed input size
mlmodel = ct.convert(
    s_model,  # use the scripted model
    source="pytorch",
    inputs=[ct.TensorType(shape=(1, 1, 1, h, w), name="input_image")],
    outputs=[ct.TensorType(name="sr_output"), ct.TensorType(name="confidence")],
    convert_to="mlprogram"
)

mlmodel.save("Bayesian_DPATISR.mlpackage")
print("🎉 FINAL SUCCESS! Package created.")
