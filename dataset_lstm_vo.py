import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

class UnderwaterSequenceVODataset(Dataset):
    def __init__(self, data_dir, seq_len=200, step_stride=100, img_size=(224, 224)):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.img_size = img_size
        self.img_dir = self.data_dir / "cam0_images"

        # 1. Load image metadata
        csv_path = self.data_dir / "images.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing images.csv in {self.data_dir}")
        self.img_df = pd.read_csv(csv_path)

        # 2. Load MoCap ground truth
        tum_files = list(self.data_dir.glob("*mocap*.tum"))
        if not tum_files:
            tum_files = list(self.data_dir.glob("*.tum"))
        if not tum_files:
            raise FileNotFoundError(f"No .tum ground truth file found in {self.data_dir}")
        
        gt_path = tum_files[0]
        self.gt_df = pd.read_csv(
            gt_path, 
            sep=r"\s+", 
            header=None,
            names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"]
        )

        # 3. Timestamp alignment
        cam_times = self.img_df["timestamp_sec"].values
        mocap_times = self.gt_df["t"].values

        t_min = max(cam_times[0], mocap_times[0])
        t_max = min(cam_times[-1], mocap_times[-1])

        valid_mask = (cam_times >= t_min) & (cam_times <= t_max)
        self.img_df = self.img_df[valid_mask].reset_index(drop=True)
        cam_times = self.img_df["timestamp_sec"].values

        # 4. Interpolate MoCap poses
        mocap_pos = self.gt_df[["x", "y", "z"]].values
        pos_interp = interp1d(mocap_times, mocap_pos, axis=0, kind="linear")
        self.interp_pos = pos_interp(cam_times)

        mocap_quats = self.gt_df[["qx", "qy", "qz", "qw"]].values
        rots = R.from_quat(mocap_quats)
        self.interp_rots = rots[np.searchsorted(mocap_times, cam_times).clip(0, len(mocap_times)-1)]

        # 5. Precompute step-to-step 6-DoF relative poses
        self.rel_poses = []
        for i in range(len(self.img_df) - 1):
            R_i = self.interp_rots[i].as_matrix()
            R_next = self.interp_rots[i + 1].as_matrix()
            p_i = self.interp_pos[i]
            p_next = self.interp_pos[i + 1]

            dp_local = R_i.T @ (p_next - p_i)
            dR = R_i.T @ R_next
            d_euler = R.from_matrix(dR).as_euler("xyz", degrees=False)
            self.rel_poses.append(np.hstack((dp_local, d_euler)).astype(np.float32))

        self.rel_poses = np.array(self.rel_poses)  # Shape: (N-1, 6)

        # 6. Build sequence window slices: each window needs seq_len images -> (seq_len - 1) step motions
        self.window_indices = []
        total_steps = len(self.rel_poses)
        target_span = seq_len - 1

        for start_idx in range(0, total_steps - target_span + 1, step_stride):
            self.window_indices.append(start_idx)

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        start_idx = self.window_indices[idx]
        img_slice_indices = range(start_idx, start_idx + self.seq_len)

        # Load consecutive images across the window
        frames = []
        for img_i in img_slice_indices:
            fname = self.img_df.iloc[img_i]["filename"]
            img_path = self.img_dir / fname
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Failed to load image: {img_path}")
            img = cv2.resize(img, self.img_size)
            # Standard ImageNet RGB normalization
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            # To tensor (C, H, W)
            frames.append(torch.from_numpy(img).permute(2, 0, 1).float())

        # Shape: (seq_len, 3, H, W)
        seq_frames = torch.stack(frames, dim=0)

        # Relative poses over the transitions: length (seq_len - 1, 6)
        seq_targets = torch.from_numpy(self.rel_poses[start_idx : start_idx + self.seq_len - 1])

        return seq_frames, seq_targets
