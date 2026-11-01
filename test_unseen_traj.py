import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from dataset_vo import UnderwaterVODataset
from train_multi_trajectory import DeepVOModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running test evaluation on: {device}")

# 1. Load unseen traj_06 dataset
test_dir = "processed_data/traj_06"
dataset = UnderwaterVODataset(data_dir=test_dir)

model = DeepVOModel(pretrained=False).to(device)
model.load_state_dict(torch.load("best_multi_traj_vo.pth", map_location=device))
model.eval()

# 2. Sequential frame-to-frame prediction
print(f"\nPredicting consecutive relative poses on unseen {test_dir}...")
predicted_rel_poses = []
with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        img_pair, _ = dataset[i]
        img_pair = img_pair.unsqueeze(0).to(device)
        pred_6dof = model(img_pair).cpu().squeeze(0).numpy()
        predicted_rel_poses.append(pred_6dof)

# 3. Compounding trajectory into global coordinates
print("Compounding trajectory into global world frame...")
initial_pos = dataset.interp_pos[0]
initial_rot = dataset.interp_rots[0].as_matrix()

curr_R = initial_rot
curr_p = initial_pos

tum_rows = []
q_init = R.from_matrix(curr_R).as_quat()
t0 = dataset.img_df.iloc[0]["timestamp_sec"]
tum_rows.append(f"{t0:.6f} {curr_p[0]:.6f} {curr_p[1]:.6f} {curr_p[2]:.6f} {q_init[0]:.6f} {q_init[1]:.6f} {q_init[2]:.6f} {q_init[3]:.6f}\n")

for i, rel in enumerate(predicted_rel_poses):
    t_step = dataset.img_df.iloc[i + 1]["timestamp_sec"]
    dp = rel[:3]
    d_euler = rel[3:]
    
    dR = R.from_euler('xyz', d_euler).as_matrix()
    
    curr_p = curr_p + curr_R @ dp
    curr_R = curr_R @ dR
    
    quat = R.from_matrix(curr_R).as_quat()
    tum_rows.append(f"{t_step:.6f} {curr_p[0]:.6f} {curr_p[1]:.6f} {curr_p[2]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")

out_file = Path("multi_traj_estimated_traj06.tum")
with open(out_file, "w") as f:
    f.writelines(tum_rows)

print(f"\nSaved estimated trajectory to: {out_file.resolve()}")
