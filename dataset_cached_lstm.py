import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

class CachedSequenceDataset(Dataset):
    def __init__(self, data_dir, seq_len=200, step_stride=100):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len

        # 1. Load precomputed ResNet latents
        latent_file = self.data_dir / "resnet_latents.pt"
        if not latent_file.exists():
            raise FileNotFoundError(f"Missing {latent_file}. Run cache_latents.py first.")
        self.latents = torch.load(latent_file, map_location="cpu")

        # 2. Metadata & ground truth
        self.img_df = pd.read_csv(self.data_dir / "images.csv")
        tum_files = list(self.data_dir.glob("*mocap*.tum"))
        if not tum_files:
            tum_files = list(self.data_dir.glob("*.tum"))
        tum_file = tum_files[0]
        
        self.gt_df = pd.read_csv(
            tum_file, 
            sep=r"\s+", 
            header=None,
            names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"]
        )

        # 3. Synchronize timestamps
        cam_times = self.img_df["timestamp_sec"].values
        mocap_times = self.gt_df["t"].values
        t_min, t_max = max(cam_times[0], mocap_times[0]), min(cam_times[-1], mocap_times[-1])
        valid_mask = (cam_times >= t_min) & (cam_times <= t_max)
        
        self.latents = self.latents[valid_mask]
        cam_times = cam_times[valid_mask]

        # 4. Interpolate MoCap
        mocap_pos = self.gt_df[["x", "y", "z"]].values
        pos_interp = interp1d(mocap_times, mocap_pos, axis=0, kind="linear")
        self.interp_pos = pos_interp(cam_times)

        mocap_quats = self.gt_df[["qx", "qy", "qz", "qw"]].values
        rots = R.from_quat(mocap_quats)
        self.interp_rots = rots[np.searchsorted(mocap_times, cam_times).clip(0, len(mocap_times)-1)]

        # 5. Precompute relative poses
        rel_poses = []
        for i in range(len(self.latents) - 1):
            R_i = self.interp_rots[i].as_matrix()
            R_next = self.interp_rots[i + 1].as_matrix()
            p_i = self.interp_pos[i]
            p_next = self.interp_pos[i + 1]

            dp_local = R_i.T @ (p_next - p_i)
            dR = R_i.T @ R_next
            d_euler = R.from_matrix(dR).as_euler("xyz", degrees=False)
            rel_poses.append(np.hstack((dp_local, d_euler)).astype(np.float32))

        self.rel_poses = torch.from_numpy(np.array(rel_poses))

        # 6. Slices
        self.indices = list(range(0, len(self.rel_poses) - (seq_len - 1) + 1, step_stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        seq_features = self.latents[start : start + self.seq_len]
        seq_targets = self.rel_poses[start : start + self.seq_len - 1]
        return seq_features, seq_targets
