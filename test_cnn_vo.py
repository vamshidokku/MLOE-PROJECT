import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from dataset_vo import UnderwaterVODataset
from train_cnn_vo import DeepVOModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on: {device}")

# 1. Load dataset sequentially
dataset = UnderwaterVODataset()
model = DeepVOModel(pretrained=False).to(device)
model.load_state_dict(torch.load("best_cnn_vo.pth", map_location=device))
model.eval()

# 2. Sequential frame-to-frame prediction
print("\nPredicting consecutive relative poses...")
predicted_rel_poses = []
with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        img_pair, _ = dataset[i]
        img_pair = img_pair.unsqueeze(0).to(device)  # shape: (1, 2, H, W)
        pred_6dof = model(img_pair).cpu().squeeze(0).numpy()
        predicted_rel_poses.append(pred_6dof)

# 3. Compounding trajectory: T_{k+1} = T_k * ΔT_k
print("Compounding trajectory into global world frame...")
initial_pos = dataset.interp_pos[0]
initial_rot = dataset.interp_rots[0].as_matrix()

curr_R = initial_rot
curr_p = initial_pos

tum_rows = []
q_init = R.from_matrix(curr_R).as_quat()  # [qx, qy, qz, qw]
t0 = dataset.img_df.iloc[0]["timestamp_sec"]
tum_rows.append(f"{t0:.6f} {curr_p[0]:.6f} {curr_p[1]:.6f} {curr_p[2]:.6f} {q_init[0]:.6f} {q_init[1]:.6f} {q_init[2]:.6f} {q_init[3]:.6f}\n")

for i, rel in enumerate(predicted_rel_poses):
    t_step = dataset.img_df.iloc[i + 1]["timestamp_sec"]
    dp = rel[:3]
    d_euler = rel[3:]
    
    dR = R.from_euler('xyz', d_euler).as_matrix()
    
    # Global update
    curr_p = curr_p + curr_R @ dp
    curr_R = curr_R @ dR
    
    quat = R.from_matrix(curr_R).as_quat()
    tum_rows.append(f"{t_step:.6f} {curr_p[0]:.6f} {curr_p[1]:.6f} {curr_p[2]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")

out_file = Path("cnn_estimated_traj.tum")
with open(out_file, "w") as f:
    f.writelines(tum_rows)

print(f"\nSaved estimated trajectory to: {out_file.resolve()}")
