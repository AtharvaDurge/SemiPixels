import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SemiconductorDataset
from models.nafnet_sr import NAFNetSR

# 1. Direct Pixel Loss for Maximum PSNR
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps2 = eps ** 2
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: CUDA not detected. Training will be extremely slow.")

    # 2. Model & Checkpoint Loading (NAFNet-SR 2x)
    model = NAFNetSR(
        img_channel=1, 
        width=32, 
        middle_blk_num=2, 
        enc_blk_nums=[2, 2, 2], 
        dec_blk_nums=[2, 2, 2], 
        scale_factor=2
    ).to(device)

    # Resume directly from your best 24.04 dB checkpoint
    ckpt_path = 'weights/model_best.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint '{ckpt_path}' not found. Please train/finetune first.")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['ema_state_dict'] if 'ema_state_dict' in ckpt else (ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=True)

    # 3. Data Loader Setup
    train_ds = SemiconductorDataset('data/train/degraded', 'data/train/gt', patch_size=128, scale=2, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    # Charbonnier L1 loss directly maximizes PSNR
    pixel_loss = CharbonnierLoss().to(device)

    # 4. Ultra-Low LR Scheduler (for fine-tuning the last 2-4 dB)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-7)
    scaler = torch.amp.GradScaler('cuda')

    epochs = 10
    best_loss = float('inf')
    os.makedirs('weights', exist_ok=True)

    print(f"--> Starting Ultra-Low LR PSNR Fine-Tuning (10 Epochs) on {device.upper()}...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for deg, gt, g_min, g_max, _ in train_loader:
            deg, gt = deg.to(device), gt.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                pred = model(deg)
                # Pure Charbonnier loss (pixel distance minimizing)
                loss = pixel_loss(pred, gt)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Keep gradients smooth at ultra-low LR
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{epochs}] | Pixel Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")
        
        # Save whenever the direct pixel loss is lower
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': model.state_dict()}, 'weights/model_best.pt')
            print(f"  [+] Saved optimized checkpoint (Pixel Loss: {avg_loss:.6f})")

    print("\nPSNR-optimized fine-tuning complete. Best checkpoint saved to weights/model_best.pt.")

if __name__ == '__main__':
    # Required guard for Windows multi-worker dataloading
    main()