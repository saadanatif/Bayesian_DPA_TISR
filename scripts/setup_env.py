#!/usr/bin/env python
import platform, shutil, subprocess, sys
import ensurepip
ensurepip.bootstrap()


TORCH = "2.1.0"
TORCHVISION = "0.16.0"
TORCHAUDIO = "2.1.0"
# Default to CUDA 12.1 wheels when NVIDIA GPU is present; otherwise CPU wheels
TORCH_INDEX = {
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

def has_nvidia():
    return shutil.which("nvidia-smi") is not None

def pip_install(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])

def main():
    pip_install("--upgrade", "pip")
    index = TORCH_INDEX["cu121"] if has_nvidia() else TORCH_INDEX["cpu"]
    pip_install(
        f"torch=={TORCH}",
        f"torchvision=={TORCHVISION}",
        f"torchaudio=={TORCHAUDIO}",
        "--index-url", index,
    )
    pip_install("openmim")
    subprocess.check_call([sys.executable, "-m", "mim", "install", f"mmcv-full==1.5.0"])
    pip_install("-r", "requirements.txt")
    if platform.system() == "Darwin":
        pip_install("coremltools==7.0")  # only works on macOS
    print("All set.")

if __name__ == "__main__":
    main()