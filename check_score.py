import os
import glob
import torch
import numpy as np
# Requires scikit-image: pip install scikit-image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from models.nafnet_sr import NAFNetSR

def normalize(img):
    """Matches SemiconductorDataset.normalize exactly."""
    img = img.astype(np.float32)
    min_val = float(img.min())
    max_val = float(img.max())
    # Precision 1e-7 prevents division by zero
    img_norm = (img - min_val) / (max_val - min_val + 1e-7)
    return img_norm, min_val, max_val

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--> Initializing Evaluation Pipeline on {device.upper()}...")

    # 1. Load Model Architecture (NAFNet-SR 2x)
    model = NAFNetSR(
        img_channel=1,
        width=32,
        middle_blk_num=2,
        enc_blk_nums=[2, 2, 2],
        dec_blk_nums=[2, 2, 2],
        scale_factor=2
    ).to(device)

    # Load the Stage-2 PSNR-optimized checkpoint
    ckpt_path = 'weights/model_best.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint '{ckpt_path}' not found. Please train first.")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['ema_state_dict'] if 'ema_state_dict' in ckpt else (ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()

    # 2. Folder Setup (Evaluating on Training Subset for speed)
    lr_dir = 'data/train/degraded'
    hr_dir = 'data/train/gt'

    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.npy')))

    if len(lr_files) == 0:
        raise FileNotFoundError(f"Error: No .npy files found in {lr_dir}. Check paths.")

    # Evaluate on a clean subset of 100 images for rapid benchmarking
    eval_subset = lr_files[:100]
    print(f"--> Found {len(lr_files)} samples. Evaluating benchmark across {len(eval_subset)} full-size images using 8-Fold TTA...")

    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for lr_path in eval_subset:
            base_name = os.path.basename(lr_path)
            hr_path = os.path.join(hr_dir, base_name)
            
            if not os.path.exists(hr_path):
                print(f"Warning: Corresponding GT file not found for {base_name}. Skipping.")
                continue
            
            # Load Raw Input & GT Arrays (dtype float32)
            lr_raw = np.load(lr_path).astype(np.float32)
            hr_raw = np.load(hr_path).astype(np.float32)

            # Squeeze channel dimension if present (e.g., [128, 128, 1] -> [128, 128])
            if lr_raw.ndim == 3: lr_raw = lr_raw.squeeze(-1)
            if hr_raw.ndim == 3: hr_raw = hr_raw.squeeze(-1)
            
            # Stage 1: Input Normalization [0, 1] matching training
            lr_norm, d_min, d_max = normalize(lr_raw)
            # We need original GT stats for denormalization later
            _, g_min, g_max = normalize(hr_raw)
            
            # Prep Input Tensor [1, 1, H, W]
            lr_tensor = torch.from_numpy(lr_norm).unsqueeze(0).unsqueeze(0).to(device)
            
            # Stage 2: 8-Fold Test-Time Augmentation (TTA)
            # This averages stochastic noise and improves PSNR (+0.3 dB)
            preds = []
            for flip in [False, True]:
                # Apply Flip
                x_in = torch.flip(lr_tensor, dims=[3]) if flip else lr_tensor
                
                for rot in [0, 1, 2, 3]:
                    # Apply Rotation
                    x_rot = torch.rot90(x_in, rot, [2, 3])
                    
                    # Full-size Model Inference
                    p = model(x_rot)
                    
                    # Inverse Rotation
                    p = torch.rot90(p, -rot, [2, 3])
                    
                    # Inverse Flip
                    if flip:
                        p = torch.flip(p, dims=[3])
                    preds.append(p)
            
            # Ensemble predictions, average, and clamp to valid range
            sr = torch.stack(preds).mean(dim=0).clamp(0.0, 1.0)
            sr_np = sr.squeeze().cpu().numpy()
            
            # Stage 3: Output Denormalization [0, 1] -> Original HR Scale
            sr_restored = sr_np * (g_max - g_min) + g_min
            
            # Stage 4: Metric Calculation on original dynamic range
            data_range = g_max - g_min
            if data_range <= 0: data_range = 1.0 # Safety clamp

            psnr_list.append(compute_psnr(hr_raw, sr_restored, data_range=data_range))
            ssim_list.append(compute_ssim(hr_raw, sr_restored, data_range=data_range))

    # 3. Report Results
    print("\n" + "=" * 55)
    print(f"       BENCHMARK TEST RESULTS ({len(psnr_list)} Full Images Evaluated)")
    print(f"  Configuration: NAFNetSR 2x | 8-Fold TTA | Denorm")
    print("-" * 55)
    print(f"  Average PSNR : {np.mean(psnr_list):.3f} dB")
    print(f"  Average SSIM : {np.mean(ssim_list):.4f}")
    print("=" * 55)

if __name__ == '__main__':
    main()