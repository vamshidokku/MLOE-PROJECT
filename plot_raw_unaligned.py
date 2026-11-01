import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Ground Truth and CNN Predicted Trajectories
gt_file = "processed_data/traj_01/traj_01_mocap.tum"
est_file = "cnn_estimated_traj.tum"

df_gt = pd.read_csv(gt_file, sep=" ", header=None, 
                    names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
df_est = pd.read_csv(est_file, sep=" ", header=None, 
                     names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

# 2. Setup 3D Plot
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot Ground Truth
ax.plot(df_gt["x"], df_gt["y"], df_gt["z"], 
        label="MoCap Ground Truth", color="black", linestyle="--", linewidth=1.8)

# Plot Raw CNN Estimation
ax.plot(df_est["x"], df_est["y"], df_est["z"], 
        label="Raw CNN Predicted (ResNet-18)", color="crimson", linewidth=1.8)

# Mark Start and End Points explicitly
ax.scatter(df_gt["x"].iloc[0], df_gt["y"].iloc[0], df_gt["z"].iloc[0], 
           color="lime", s=90, edgecolors='k', label="Shared Start Point (t=0)")

ax.scatter(df_gt["x"].iloc[-1], df_gt["y"].iloc[-1], df_gt["z"].iloc[-1], 
           color="blue", s=80, marker='^', label="GT End Point")

ax.scatter(df_est["x"].iloc[-1], df_est["y"].iloc[-1], df_est["z"].iloc[-1], 
           color="red", s=80, marker='x', label="CNN Drifted End Point")

ax.set_title("Raw Unaligned Trajectory Comparison (Starting at Identical Pose)")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig("raw_unaligned_comparison.png", dpi=300)
print("Saved raw comparison plot to 'raw_unaligned_comparison.png'. Displaying window...")
plt.show()
