import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.nafnet_sr import NAFNetSR
from dataset import SemiconductorDataset

# --- LOSS FUNCTIONS ---

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + (self.eps * self.eps))
        return torch.mean(loss)


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)

        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        return loss_x + loss_y


# --- MAIN TRAINING LOOP ---

def train():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    DEGRADED_DIR = os.path.join(BASE_DIR, "data", "train", "degraded")
    GT_DIR = os.path.join(BASE_DIR, "data", "train", "gt")
    SAVE_DIR = os.path.join(BASE_DIR, "weights")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Hyperparameters
    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 2e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on device: {DEVICE}")

    train_dataset = SemiconductorDataset(degraded_dir=DEGRADED_DIR, gt_dir=GT_DIR, is_train=True)
    print(f"Successfully loaded {len(train_dataset)} training samples!")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = NAFNetSR(img_channel=1, scale_factor=2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    criterion_charbonnier = CharbonnierLoss().to(DEVICE)
    criterion_sobel = SobelEdgeLoss().to(DEVICE)

    # Modern AMP API to prevent deprecation warnings
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))

    model.train()
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0

        for batch_idx, (deg, gt, _) in enumerate(train_loader):
            deg, gt = deg.to(DEVICE), gt.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                output = model(deg)
                loss_charb = criterion_charbonnier(output, gt)
                loss_sobel = criterion_sobel(output, gt)
                loss = loss_charb + (0.5 * loss_sobel)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{EPOCHS}] - Loss: {epoch_loss:.6f} - LR: {scheduler.get_last_lr()[0]:.6f}")

    final_model_path = os.path.join(SAVE_DIR, "model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete! Final model saved to: {final_model_path}")

if __name__ == "__main__":
    train()