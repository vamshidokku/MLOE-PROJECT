import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evo.tools import file_interface
from evo.core import sync, metrics
from evo.core.trajectory import PoseTrajectory3D

# 1. Load TUM trajectories
gt_file = "processed_data/traj_01/traj_01_mocap.tum"
est_file = "cnn_estimated_traj.tum"

print("Loading trajectories...")
traj_ref = file_interface.read_tum_trajectory_file(gt_file)
traj_est = file_interface.read_tum_trajectory_file(est_file)

# 2. Synchronize timestamps (max difference tolerance 0.05s)
traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.05)

# 3. Perform SE(3) Umeyama alignment
traj_est.align(traj_ref, correct_scale=False)

# 4. Compute Absolute Pose Error (APE) translation RMSE
data = (traj_ref, traj_est)
ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
ape_metric.process_data(data)
ape_stats = ape_metric.get_all_statistics()

print("\n" + "="*45)
print("     ABSOLUTE POSE ERROR (APE) RESULTS")
print("="*45)
for k, v in ape_stats.items():
    print(f"{k.upper():<10} : {v:.4f} m")
print("="*45)

# 5. Plot Comparison
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(traj_ref.positions_xyz[:, 0], traj_ref.positions_xyz[:, 1], traj_ref.positions_xyz[:, 2], 
        label="MoCap Ground Truth", color="black", linestyle="--", linewidth=1.5)
ax.plot(traj_est.positions_xyz[:, 0], traj_est.positions_xyz[:, 1], traj_est.positions_xyz[:, 2], 
        label="CNN Predicted (ResNet-18)", color="crimson", linewidth=1.5)

ax.set_title(f"Trajectory 01: Ground Truth vs CNN VO\nATE RMSE: {ape_stats['rmse']:.3f} m")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.legend()
plt.tight_layout()
plt.savefig("cnn_trajectory_comparison.png", dpi=300)
print("\nPlot saved as 'cnn_trajectory_comparison.png'. Displaying plot window...")
plt.show()
