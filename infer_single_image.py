# Single image inference script for Bayesian DPA-TISR
# Scales a single image 3x using the pre-trained model

import argparse
import torch
import numpy as np
from PIL import Image
import yaml


def load_model(config_path, device):
    """Load the DPATISR model with checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    from model_2D.models.backbones.sr_backbones import DPATISR
    
    model = DPATISR(
        mid_channels=config['mid_channels'],
        extraction_nblocks=config['extraction_nblocks'],
        propagation_nblocks=config['propagation_nblocks'],
        reconstruction_nblocks=config['reconstruction_nblocks'],
        factor=config['factor'],
        bayesian=config['bayesian']
    ).to(device)
    
    checkpoint = torch.load(config['inference_checkpt'], map_location=device)
    
    # Handle DataParallel checkpoint keys
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    return model, config


def preprocess_image(image_path, patch_size=128):
    """Load and preprocess a single image for the model."""
    img = Image.open(image_path)
    
    # Convert to grayscale if needed (model expects single channel)
    if img.mode != 'L':
        img = img.convert('L')
    
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize to 0-1 range
    img_array = img_array / 255.0
    
    # Pad to multiple of patch_size for proper processing
    h, w = img_array.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    if pad_h > 0 or pad_w > 0:
        img_array = np.pad(img_array, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    return img_array, (h, w)


def create_temporal_sequence(img_array, num_frames=7):
    """Create a temporal sequence by replicating the single image."""
    # Model expects shape (n, t, c, h, w)
    # Replicate single frame to create a pseudo-temporal sequence
    h, w = img_array.shape
    sequence = np.tile(img_array, (num_frames, 1, 1))
    sequence = sequence.reshape(1, num_frames, 1, h, w)
    return sequence


def run_inference(model, img_tensor, device, bayesian=True):
    """Run model inference on the image tensor."""
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        
        # Take the middle frame (index 3 for 7 frames)
        middle_idx = output.shape[1] // 2
        result = output[:, middle_idx, 0:1, :, :]  # Take only the SR output, not uncertainty
        
        return result.squeeze().cpu().numpy()


def postprocess_and_save(result, original_size, output_path, scale_factor=3):
    """Post-process and save the super-resolved image."""
    h, w = original_size
    target_h, target_w = h * scale_factor, w * scale_factor
    
    # Crop to original size * scale_factor
    result = result[:target_h, :target_w]
    
    # Clip values to valid range and convert to uint8
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    # Save the result
    output_img = Image.fromarray(result, mode='L')
    output_img.save(output_path)
    print(f"Saved super-resolved image to: {output_path}")
    
    return output_img


def main():
    parser = argparse.ArgumentParser(description='Scale a single image 3x using Bayesian DPA-TISR')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, config = load_model(args.config, device)
    scale_factor = config['factor']
    print(f"Scale factor: {scale_factor}x")
    
    # Preprocess image
    print(f"Processing image: {args.input}")
    img_array, original_size = preprocess_image(args.input)
    
    # Create temporal sequence (model expects multiple frames)
    sequence = create_temporal_sequence(img_array, num_frames=7)
    img_tensor = torch.from_numpy(sequence).float()
    
    # Run inference
    print("Running super-resolution...")
    result = run_inference(model, img_tensor, device, config['bayesian'])
    
    # Save result
    postprocess_and_save(result, original_size, args.output, scale_factor)
    print("Done!")


if __name__ == '__main__':
    main()
