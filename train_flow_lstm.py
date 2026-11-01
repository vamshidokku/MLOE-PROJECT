import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

class FlowSequenceDataset(Dataset):
    def __init__(self, data_dir, seq_len=200, step_stride=50):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len

        resnet_latents = torch.load(self.data_dir / "resnet_latents.pt", map_location="cpu")
        flow_latents = torch.load(self.data_dir / "flow_latents.pt", map_location="cpu")
        self.features = torch.cat([resnet_latents, flow_latents], dim=-1)

        self.img_df = pd.read_csv(self.data_dir / "images.csv")
        tum_files = list(self.data_dir.glob("*mocap*.tum"))
        if not tum_files:
            tum_files = list(self.data_dir.glob("*.tum"))
        self.gt_df = pd.read_csv(tum_files[0], sep=r"\s+", header=None, names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

        cam_times = self.img_df["timestamp_sec"].values
        mocap_times = self.gt_df["t"].values
        t_min, t_max = max(cam_times[0], mocap_times[0]), min(cam_times[-1], mocap_times[-1])
        valid_mask = (cam_times >= t_min) & (cam_times <= t_max)
        
        self.features = self.features[valid_mask]
        cam_times = cam_times[valid_mask]

        mocap_pos = self.gt_df[["x", "y", "z"]].values
        pos_interp = interp1d(mocap_times, mocap_pos, axis=0, kind="linear")
        interp_pos = pos_interp(cam_times)

        mocap_quats = self.gt_df[["qx", "qy", "qz", "qw"]].values
        rots = R.from_quat(mocap_quats)
        interp_rots = rots[np.searchsorted(mocap_times, cam_times).clip(0, len(mocap_times)-1)]

        rel_poses = []
        for i in range(len(self.features) - 1):
            R_i = interp_rots[i].as_matrix()
            R_next = interp_rots[i + 1].as_matrix()
            p_i = interp_pos[i]
            p_next = interp_pos[i + 1]

            dp_local = R_i.T @ (p_next - p_i)
            dR = R_i.T @ R_next
            d_euler = R.from_matrix(dR).as_euler("xyz", degrees=False)
            rel_poses.append(np.hstack((dp_local, d_euler)).astype(np.float32))

        self.rel_poses = torch.from_numpy(np.array(rel_poses))
        self.indices = list(range(0, len(self.rel_poses) - (seq_len - 1) + 1, step_stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        seq_feats = self.features[start : start + self.seq_len]
        seq_targets = self.rel_poses[start : start + self.seq_len - 1]
        return seq_feats, seq_targets

class FlowLSTMVO(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_layers=2):
        super(FlowLSTMVO, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        step_outputs = lstm_out[:, :-1, :]
        return self.fc(step_outputs)

class PoseLoss(nn.Module):
    def __init__(self, rot_weight=200.0):
        super(PoseLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.rot_weight = rot_weight

    def forward(self, pred, target):
        loss_pos = self.mse(pred[:, :, :3], target[:, :, :3])
        loss_rot = self.mse(pred[:, :, 3:], target[:, :, 3:])
        return loss_pos + self.rot_weight * loss_rot

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Flow-Enhanced LSTM on: {device}")

    train_trajs = ["traj_01", "traj_02", "traj_03", "traj_04", "traj_05"]
    train_ds = ConcatDataset([FlowSequenceDataset(f"processed_data/{t}", seq_len=200, step_stride=50) for t in train_trajs])
    val_ds = FlowSequenceDataset("processed_data/traj_06", seq_len=200, step_stride=100)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = FlowLSTMVO(input_dim=1024, hidden_dim=256, num_layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)
    criterion = PoseLoss(rot_weight=200.0)

    epochs = 40
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for seq_feats, seq_targets in train_loader:
            seq_feats, seq_targets = seq_feats.to(device), seq_targets.to(device)
            optimizer.zero_grad()
            preds = model(seq_feats)
            loss = criterion(preds, seq_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train_loss += loss.item() * seq_feats.size(0)

        epoch_train = running_train_loss / len(train_ds)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for seq_feats, seq_targets in val_loader:
                seq_feats, seq_targets = seq_feats.to(device), seq_targets.to(device)
                preds = model(seq_feats)
                loss = criterion(preds, seq_targets)
                running_val_loss += loss.item() * seq_feats.size(0)

        epoch_val = running_val_loss / len(val_ds)
        scheduler.step(epoch_val)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch_val < best_val_loss:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train:.6f} | Val Loss: {epoch_val:.6f}")

        if epoch_val < best_val_loss:
            best_val_loss = epoch_val
            torch.save(model.state_dict(), "best_flow_lstm_vo.pth")

    print("\nTraining complete! Saved to 'best_flow_lstm_vo.pth'")

if __name__ == "__main__":
    main()
