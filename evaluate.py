import os
import glob
import argparse
import cv2
import torch
import numpy as np
from models.nafnet_sr import NAFNetSR

def parse_args():
    parser = argparse.ArgumentParser(description="KLA Benchmark Fast Inference")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input test directory (.npy)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--weights", type=str, default="weights/model_best.pt", help="Path to checkpoint weights")
    parser.add_argument("--tta", action="store_true", help="Enable 8-Fold Test-Time Augmentation")
    return parser.parse_args()

def forward_tta(model, tensor):
    """8-Fold Test-Time Augmentation (4 rotations x 2 flips)"""
    preds = []
    for flip in [False, True]:
        x_in = torch.flip(tensor, dims=[3]) if flip else tensor
        for rot in [0, 1, 2, 3]:
            x_rot = torch.rot90(x_in, rot, [2, 3])
            p = model(x_rot)
            p = torch.rot90(p, -rot, [2, 3])
            if flip:
                p = torch.flip(p, dims=[3])
            preds.append(p)
    return torch.stack(preds).mean(dim=0)

@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Initializing Inference Pipeline on {device.type.upper()}...")

    # 1. Initialize NAFNet-SR (width=32)
    model = NAFNetSR(
        img_channel=1,
        width=32,
        middle_blk_num=2,
        enc_blk_nums=[2, 2, 2],
        dec_blk_nums=[2, 2, 2],
        scale_factor=2
    ).to(device)

    # 2. Robust Checkpoint Loading
    weights_path = args.weights if os.path.exists(args.weights) else "weights/model_final.pt"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at '{weights_path}'.")

    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict):
        if "ema_state_dict" in ckpt:
            state_dict = ckpt["ema_state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=True)
    print(f"--> Loaded weights from: {weights_path}")

    model.eval()

    # 3. Process Input Files
    file_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not file_paths:
        file_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.*")))
    print(f"--> Found {len(file_paths)} test files in '{args.input_dir}'. Running inference (TTA={args.tta})...")

    for file_path in file_paths:
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        if file_path.lower().endswith(".npy"):
            raw_img = np.load(file_path).astype(np.float32)
        else:
            raw_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        if raw_img.ndim == 3:
            raw_img = raw_img.squeeze(-1)

        d_min, d_max = float(raw_img.min()), float(raw_img.max())
        img_norm = (raw_img - d_min) / (d_max - d_min + 1e-7)

        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            if args.tta:
                output = forward_tta(model, tensor)
            else:
                output = model(tensor)

        out_img = output.squeeze().clamp(0.0, 1.0).cpu().numpy()
        restored_float = out_img * (d_max - d_min) + d_min
        restored_uint8 = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)

        # Save both full-precision .npy array and visualization .png
        np.save(os.path.join(args.output_dir, f"{base_name}.npy"), restored_float)
        cv2.imwrite(os.path.join(args.output_dir, f"{base_name}.png"), restored_uint8)

    print(f"--> Inference finished successfully. Results saved to '{args.output_dir}'.")

if __name__ == "__main__":
    main()