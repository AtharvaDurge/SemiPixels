import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from models.nafnet_sr import NAFNetSR
from dataset import SemiconductorDataset

# --- HIGH-SPEED LOSS FUNCTIONS ---

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))

class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        px = F.conv2d(pred, self.sobel_x, padding=1)
        py = F.conv2d(pred, self.sobel_y, padding=1)
        tx = F.conv2d(target, self.sobel_x, padding=1)
        ty = F.conv2d(target, self.sobel_y, padding=1)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)

class FastSSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        sigma = 1.5
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        gauss = (gauss / gauss.sum()).unsqueeze(1)
        _2D_window = (gauss @ gauss.t()).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', _2D_window)

    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-7)
        return 1.0 - ssim_map.mean()

class FastCompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.edge = SobelEdgeLoss()
        self.ssim = FastSSIMLoss()

    def forward(self, pred, target):
        l_charb = self.charb(pred, target)
        l_edge = self.edge(pred, target)
        l_ssim = self.ssim(pred, target)
        return l_charb + 0.35 * l_edge + 0.25 * l_ssim

# --- TRAINING UTILITIES ---

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(1.0 / (mse + 1e-10))

def train():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEGRADED_DIR = os.path.join(BASE_DIR, "data", "train", "degraded")
    GT_DIR = os.path.join(BASE_DIR, "data", "train", "gt")
    SAVE_DIR = os.path.join(BASE_DIR, "weights")
    os.makedirs(SAVE_DIR, exist_ok=True)

    BATCH_SIZE = 16
    EPOCHS = 25
    BASE_LR = 6e-4
    WARMUP_EPOCHS = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--> Initializing 2-Hour High-Speed Training Pipeline on {DEVICE}")

    full_dataset = SemiconductorDataset(degraded_dir=DEGRADED_DIR, gt_dir=GT_DIR, patch_size=128, is_train=True)
    val_size = max(int(len(full_dataset) * 0.05), 1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = NAFNetSR(img_channel=1, width=32, middle_blk_num=2, enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2], scale_factor=2).to(DEVICE)
    ema = ModelEMA(model, decay=0.999)

    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    criterion = FastCompositeLoss().to(DEVICE)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))

    best_val_psnr = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        if epoch <= WARMUP_EPOCHS:
            lr = BASE_LR * (epoch / WARMUP_EPOCHS)
        else:
            progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            lr = 1e-6 + 0.5 * (BASE_LR - 1e-6) * (1 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for deg, gt, _, _, _ in train_loader:
            deg, gt = deg.to(DEVICE), gt.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                output = model(deg)
                loss = criterion(output, gt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            total_loss += loss.item()

        ema.ema_model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for deg, gt, _, _, _ in val_loader:
                deg, gt = deg.to(DEVICE), gt.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                    pred = ema.ema_model(deg)
                val_psnr += calculate_psnr(pred, gt).item()

        avg_loss = total_loss / len(train_loader)
        avg_val_psnr = val_psnr / len(val_loader)
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Loss: {avg_loss:.5f} | Val PSNR: {avg_val_psnr:.2f} dB | LR: {lr:.2e}")

        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save(ema.ema_model.state_dict(), os.path.join(SAVE_DIR, "model_best.pt"))
            print(f"  [+] Saved new best model checkpoint ({best_val_psnr:.2f} dB)")

    torch.save(ema.ema_model.state_dict(), os.path.join(SAVE_DIR, "model_final.pt"))
    print("--> Training completed successfully!")

if __name__ == "__main__":
    train()