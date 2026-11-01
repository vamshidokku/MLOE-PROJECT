import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from fast_train_lstm import PureLSTMVO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_windowed_inference(traj_name, model, out_filename, window_size=200):
    data_dir = Path("processed_data") / traj_name
    latent_file = data_dir / "resnet_latents.pt"
    img_df = pd.read_csv(data_dir / "images.csv")
    
    tum_file = list(data_dir.glob("*mocap*.tum"))[0]
    gt_df = pd.read_csv(tum_file, sep=r"\s+", header=None, names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

    cam_times = img_df["timestamp_sec"].values
    mocap_times = gt_df["t"].values
    t_min, t_max = max(cam_times[0], mocap_times[0]), min(cam_times[-1], mocap_times[-1])
    valid_mask = (cam_times >= t_min) & (cam_times <= t_max)

    latents = torch.load(latent_file, map_location=device)[valid_mask]
    img_df = img_df[valid_mask].reset_index(drop=True)

    q0 = gt_df.iloc[0][["qx", "qy", "qz", "qw"]].values
    p0 = gt_df.iloc[0][["x", "y", "z"]].values
    t0 = img_df.iloc[0]["timestamp_sec"]

    print(f"Running windowed inference ({window_size} frames per chunk) for {traj_name}...")

    pred_rel_list = []
    with torch.no_grad():
        for start in range(0, len(latents) - 1, window_size - 1):
            chunk = latents[start : min(start + window_size, len(latents))].unsqueeze(0)
            if chunk.shape[1] < 2:
                break
            out = model(chunk).squeeze(0).cpu().numpy()
            pred_rel_list.append(out)

    pred_rel = np.vstack(pred_rel_list)

    curr_R = R.from_quat(q0).as_matrix()
    curr_p = p0.copy()

    tum_rows = [f"{t0:.6f} {curr_p[0]:.6f} {curr_p[1]:.6f} {curr_p[2]:.6f} {q0[0]:.6f} {q0[1]:.6f} {q0[2]:.6f} {q0[3]:.6f}\n"]

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

if __name__ == "__main__":
    model = PureLSTMVO(input_dim=512, hidden_dim=256, num_layers=2).to(device)
    model.load_state_dict(torch.load("best_pure_lstm_vo.pth", map_location=device))
    model.eval()

    run_windowed_inference("traj_06", model, "lstm_windowed_traj06.tum")
    run_windowed_inference("traj_01", model, "lstm_windowed_traj01.tum")
