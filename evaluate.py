import os
import glob
import argparse
import cv2
import torch
import numpy as np
from models.nafnet_sr import NAFNetSR

def parse_args():
    parser = argparse.ArgumentParser(description="KLA Benchmark Fast Inference")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, default="weights/model_best.pt")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    return parser.parse_args()

def forward_tta(model, tensor):
    transforms = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[-1]),
        lambda x: torch.flip(x, dims=[-2]),
        lambda x: torch.rot90(x, 1, [-2, -1]),
    ]
    inv_transforms = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[-1]),
        lambda x: torch.flip(x, dims=[-2]),
        lambda x: torch.rot90(x, -1, [-2, -1]),
    ]

    preds = []
    for t, inv_t in zip(transforms, inv_transforms):
        aug_in = t(tensor)
        out = model(aug_in)
        preds.append(inv_t(out))
    return torch.stack(preds).mean(dim=0)

@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NAFNetSR(img_channel=1, width=32, middle_blk_num=2, enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2], scale_factor=2).to(device)

    weights_path = args.weights if os.path.exists(args.weights) else "weights/model_final.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded weights: {weights_path}")

    model.eval()
    file_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    print(f"Evaluating {len(file_paths)} test files...")

    for npy_path in file_paths:
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        raw_img = np.load(npy_path).astype(np.float32)
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

        np.save(os.path.join(args.output_dir, f"{base_name}.npy"), restored_float)
        cv2.imwrite(os.path.join(args.output_dir, f"{base_name}.png"), restored_uint8)

    print("Inference finished.")

if __name__ == "__main__":
    main()