import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evo.tools import file_interface
from evo.core import sync, metrics
from train_flow_lstm import FlowSequenceDataset, FlowLSTMVO
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_traj(gt_file, est_file, traj_title, ax):
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.05)
    traj_est.align_origin(traj_ref)

    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))
    stats = ape_metric.get_all_statistics()

    print(f"\n{'='*20} {traj_title} {'='*20}")
    for k in ["rmse", "mean", "median", "max"]:
        print(f"{k.upper():<8} : {stats[k]:.4f} m")

    ax.plot(traj_ref.positions_xyz[:, 0], traj_ref.positions_xyz[:, 1], traj_ref.positions_xyz[:, 2], 
            label="MoCap GT", color="black", linestyle="--", linewidth=1.5)
    ax.plot(traj_est.positions_xyz[:, 0], traj_est.positions_xyz[:, 1], traj_est.positions_xyz[:, 2], 
            label="Flow+LSTM Predicted", color="crimson", linewidth=1.5)
    ax.scatter(traj_ref.positions_xyz[0, 0], traj_ref.positions_xyz[0, 1], traj_ref.positions_xyz[0, 2],
               color="lime", s=80, edgecolors="k", label="Origin (t=0)")

    ax.set_title(f"{traj_title}\nATE RMSE: {stats['rmse']:.3f} m")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()

# 1. 3D Trajectory Comparison
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121, projection='3d')
evaluate_traj("processed_data/traj_06/traj_06_mocap.tum", "flow_lstm_traj06.tum", "Unseen traj_06 (Flow+LSTM)", ax1)

ax2 = fig.add_subplot(122, projection='3d')
evaluate_traj("processed_data/traj_01/traj_01_mocap.tum", "flow_lstm_traj01.tum", "Seen traj_01 (Flow+LSTM)", ax2)

plt.tight_layout()
plt.savefig("flow_lstm_comparison.png", dpi=300)

# 2. Step Error Breakdown on Unseen traj_06
ds_val = FlowSequenceDataset("processed_data/traj_06", seq_len=200, step_stride=200)
model = FlowLSTMVO(input_dim=1024, hidden_dim=256, num_layers=2).to(device)
model.load_state_dict(torch.load("best_flow_lstm_vo.pth", map_location=device))
model.eval()

seq_feats, seq_targets = ds_val[0]
with torch.no_grad():
    pred_window = model(seq_feats.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
target_window = seq_targets.numpy()

fig2, axs = plt.subplots(2, 3, figsize=(14, 7))
labels = ['dx (m)', 'dy (m)', 'dz (m)', 'droll (rad)', 'dpitch (rad)', 'dyaw (rad)']

for idx, ax in enumerate(axs.flat):
    ax.plot(target_window[:, idx], label="Ground Truth", color="black", alpha=0.8)
    ax.plot(pred_window[:, idx], label="Flow+LSTM Predicted", color="crimson", alpha=0.8)
    ax.set_title(f"traj_06 Window 0: {labels[idx]}")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("flow_lstm_step_breakdown.png", dpi=300)
print("\nGenerated plots: 'flow_lstm_comparison.png' and 'flow_lstm_step_breakdown.png'. Displaying...")
plt.show()
