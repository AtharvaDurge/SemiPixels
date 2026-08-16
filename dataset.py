import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class SemiconductorDataset(Dataset):
    def __init__(self, degraded_dir, gt_dir=None, patch_size=128, scale=2, is_train=True):
        super().__init__()
        self.degraded_dir = degraded_dir
        self.gt_dir = gt_dir
        self.patch_size = patch_size
        self.scale = scale
        self.is_train = is_train

        self.degraded_paths = sorted(glob.glob(os.path.join(degraded_dir, "*.npy")))

        if self.gt_dir is not None:
            self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
            assert len(self.degraded_paths) == len(self.gt_paths), \
                f"Mismatch: Found {len(self.degraded_paths)} degraded files and {len(self.gt_paths)} GT files."
        else:
            self.gt_paths = None

    def __len__(self):
        return len(self.degraded_paths)

    @staticmethod
    def normalize(img):
        img = img.astype(np.float32)
        min_val = float(img.min())
        max_val = float(img.max())
        img_norm = (img - min_val) / (max_val - min_val + 1e-7)
        return img_norm, min_val, max_val

    def _augment(self, deg_patch, gt_patch):
        if random.random() > 0.5:
            deg_patch = np.fliplr(deg_patch)
            gt_patch = np.fliplr(gt_patch)
        if random.random() > 0.5:
            deg_patch = np.flipud(deg_patch)
            gt_patch = np.flipud(gt_patch)
        k = random.randint(0, 3)
        if k > 0:
            deg_patch = np.rot90(deg_patch, k)
            gt_patch = np.rot90(gt_patch, k)
        return deg_patch.copy(), gt_patch.copy()

    def __getitem__(self, index):
        deg_path = self.degraded_paths[index]
        deg_raw = np.load(deg_path).astype(np.float32)
        
        if deg_raw.ndim == 3:
            deg_raw = deg_raw.squeeze(-1)

        deg_norm, d_min, d_max = self.normalize(deg_raw)

        if self.gt_paths is not None:
            gt_path = self.gt_paths[index]
            gt_raw = np.load(gt_path).astype(np.float32)
            if gt_raw.ndim == 3:
                gt_raw = gt_raw.squeeze(-1)

            gt_norm, g_min, g_max = self.normalize(gt_raw)

            if self.is_train and self.patch_size is not None:
                h_deg, w_deg = deg_norm.shape
                ps_deg = min(self.patch_size, h_deg, w_deg)
                ps_gt = ps_deg * self.scale

                top_d = random.randint(0, h_deg - ps_deg)
                left_d = random.randint(0, w_deg - ps_deg)
                top_g = top_d * self.scale
                left_g = left_d * self.scale

                deg_norm = deg_norm[top_d:top_d + ps_deg, left_d:left_d + ps_deg]
                gt_norm = gt_norm[top_g:top_g + ps_gt, left_g:left_g + ps_gt]

                deg_norm, gt_norm = self._augment(deg_norm, gt_norm)

            deg_tensor = torch.from_numpy(deg_norm).unsqueeze(0).float()
            gt_tensor = torch.from_numpy(gt_norm).unsqueeze(0).float()

            return deg_tensor, gt_tensor, g_min, g_max, os.path.basename(deg_path)

        deg_tensor = torch.from_numpy(deg_norm).unsqueeze(0).float()
        return deg_tensor, d_min, d_max, os.path.basename(deg_path)