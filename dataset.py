import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class SemiconductorDataset(Dataset):
    def __init__(self, degraded_dir, gt_dir=None, is_train=True):
        super().__init__()
        self.degraded_dir = degraded_dir
        self.gt_dir = gt_dir
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

    def normalize_intensity(self, img):
        # Force float32 precision
        img = img.astype(np.float32)
        p1, p99 = np.percentile(img, (0.1, 99.9))
        img = np.clip(img, p1, p99)
        img = (img - p1) / (p99 - p1 + 1e-6)
        return img.astype(np.float32), p1, p99

    def __getitem__(self, index):
        deg_path = self.degraded_paths[index]
        
        # Load raw numpy array as float32
        deg_img = np.load(deg_path).astype(np.float32)
        deg_norm, _, _ = self.normalize_intensity(deg_img)
        
        if deg_norm.ndim == 2:
            deg_tensor = torch.from_numpy(deg_norm).unsqueeze(0)
        else:
            deg_tensor = torch.from_numpy(deg_norm).permute(2, 0, 1)

        if self.gt_paths is not None:
            gt_path = self.gt_paths[index]
            gt_img = np.load(gt_path).astype(np.float32)
            
            gt_norm, _, _ = self.normalize_intensity(gt_img)
            if gt_norm.ndim == 2:
                gt_tensor = torch.from_numpy(gt_norm).unsqueeze(0)
            else:
                gt_tensor = torch.from_numpy(gt_norm).permute(2, 0, 1)

            return deg_tensor, gt_tensor, os.path.basename(deg_path)

        return deg_tensor, os.path.basename(deg_path)


if __name__ == "__main__":
    print("dataset.py initialized with float32 casting.")