import os
import argparse
import glob
import cv2
import torch
import numpy as np
from models.nafnet_sr import NAFNetSR

def parse_args():
    parser = argparse.ArgumentParser(description="KLA Image Restoration Evaluation")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing degraded test files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save restored outputs")
    parser.add_argument("--weights", type=str, default="weights/model_final.pt", help="Path to trained model weights")
    return parser.parse_args()

def preprocess(npy_path):
    img = np.load(npy_path).astype(np.float32)
    p1, p99 = np.percentile(img, (0.1, 99.9))
    img = np.clip(img, p1, p99)
    img_norm = (img - p1) / (p99 - p1 + 1e-6)
    
    if img_norm.ndim == 2:
        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
    else:
        tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
        
    return tensor, p1, p99

def postprocess(tensor, p1, p99):
    img = tensor.squeeze().cpu().numpy()
    img = img * (p99 - p1 + 1e-6) + p1
    return np.clip(img, 0, 255).astype(np.uint8), img

@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    model = NAFNetSR(img_channel=1, scale_factor=2).to(device)
    
    if os.path.exists(args.weights):
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded weights from {args.weights}")
    else:
        print(f"Warning: Weights file '{args.weights}' not found. Using untrained initialization.")
        
    model.eval()

    file_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    print(f"Found {len(file_paths)} .npy test files in '{args.input_dir}'")

    for npy_path in file_paths:
        filename = os.path.basename(npy_path)
        base_name = os.path.splitext(filename)[0]
        
        input_tensor, p1, p99 = preprocess(npy_path)
        input_tensor = input_tensor.to(device)

        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output_tensor = model(input_tensor)
        else:
            output_tensor = model(input_tensor)

        restored_uint8, restored_float = postprocess(output_tensor, p1, p99)
        
        # Save both .npy matrix output and viewable .png image
        np.save(os.path.join(args.output_dir, f"{base_name}.npy"), restored_float)
        cv2.imwrite(os.path.join(args.output_dir, f"{base_name}.png"), restored_uint8)

    print(f"Done! Restored outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()