import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from fast_train_lstm import PureLSTMVO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_trajectory_inference(traj_name, model, out_filename):
    data_dir = Path("processed_data") / traj_name
    latent_file = data_dir / "resnet_latents.pt"
    img_df = pd.read_csv(data_dir / "images.csv")
    
    tum_files = list(data_dir.glob("*mocap*.tum"))
    if not tum_files:
        tum_files = list(data_dir.glob("*.tum"))
    gt_df = pd.read_csv(
        tum_files[0], 
        sep=r"\s+", 
        header=None, 
        names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"]
    )

    cam_times = img_df["timestamp_sec"].values
    mocap_times = gt_df["t"].values
    t_min, t_max = max(cam_times[0], mocap_times[0]), min(cam_times[-1], mocap_times[-1])
    valid_mask = (cam_times >= t_min) & (cam_times <= t_max)

    latents = torch.load(latent_file, map_location=device)[valid_mask]
    img_df = img_df[valid_mask].reset_index(drop=True)

    q0 = gt_df.iloc[0][["qx", "qy", "qz", "qw"]].values
    p0 = gt_df.iloc[0][["x", "y", "z"]].values
    t0 = img_df.iloc[0]["timestamp_sec"]

    print(f"Predicting continuous sequence for {traj_name} ({len(latents)} frames)...")

    with torch.no_grad():
        x = latents.unsqueeze(0)
        pred_rel = model(x).squeeze(0).cpu().numpy()

    curr_R = R.from_quat(q0).as_matrix()
    curr_p = p0.copy()

    tum_rows = []
    tum_rows.append(f"{t0:.6f} {curr_p[0]:.6f} {curr_p[1]:.6f} {curr_p[2]:.6f} {q0[0]:.6f} {q0[1]:.6f} {q0[2]:.6f} {q0[3]:.6f}\n")

    for i in range(len(pred_rel)):
        t_step = img_df.iloc[i + 1]["timestamp_sec"]
        dp = pred_rel[i, :3]
        d_euler = pred_rel[i, 3:]

        dR = R.from_euler('xyz', d_euler).as_matrix()
        curr_p = curr_p + curr_R @ dp
        curr_R = curr_R @ dR

        quat = R.from_matrix(curr_R).as_quat()
        tum_rows.append(f"{t_step:.6f} {curr_p[0]:.6f} {curr_p[1]:.6f} {curr_p[2]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")

    with open(out_filename, "w") as f:
        f.writelines(tum_rows)
    print(f"Saved: {out_filename}")

def main():
    model = PureLSTMVO(input_dim=512, hidden_dim=256, num_layers=2).to(device)
    model.load_state_dict(torch.load("best_pure_lstm_vo.pth", map_location=device))
    model.eval()

    run_trajectory_inference("traj_06", model, "lstm_estimated_traj06.tum")
    run_trajectory_inference("traj_01", model, "lstm_estimated_traj01.tum")

if __name__ == "__main__":
    main()
