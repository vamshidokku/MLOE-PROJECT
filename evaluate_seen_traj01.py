import matplotlib.pyplot as plt
from evo.tools import file_interface
from evo.core import sync, metrics

gt_file = "processed_data/traj_01/traj_01_mocap.tum"
est_file = "multi_traj_estimated_traj01.tum"

# 1. Load trajectories
traj_ref = file_interface.read_tum_trajectory_file(gt_file)
traj_est = file_interface.read_tum_trajectory_file(est_file)

# 2. Synchronize timestamps
traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.05)

# 3. Origin alignment (exact same start point)
traj_est.align_origin(traj_ref)

# 4. Compute Absolute Pose Error
ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
ape_metric.process_data((traj_ref, traj_est))
ape_stats = ape_metric.get_all_statistics()

print("\n" + "="*45)
print("     SEEN TRAJ_01 EVALUATION RESULTS")
print("="*45)
for k, v in ape_stats.items():
    print(f"{k.upper():<10} : {v:.4f} m")
print("="*45)

# 5. Plot Comparison
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(traj_ref.positions_xyz[:, 0], traj_ref.positions_xyz[:, 1], traj_ref.positions_xyz[:, 2], 
        label="MoCap Ground Truth (traj_01)", color="black", linestyle="--", linewidth=1.5)
ax.plot(traj_est.positions_xyz[:, 0], traj_est.positions_xyz[:, 1], traj_est.positions_xyz[:, 2], 
        label="Multi-Traj DeepVO", color="crimson", linewidth=1.5)

ax.scatter(traj_ref.positions_xyz[0, 0], traj_ref.positions_xyz[0, 1], traj_ref.positions_xyz[0, 2],
           color="lime", s=90, edgecolors="k", label="Shared Start Point (t=0)")

ax.set_title(f"Seen Trajectory 01 (Origin Aligned)\nATE RMSE: {ape_stats['rmse']:.3f} m")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.legend()
plt.tight_layout()
plt.savefig("seen_traj01_comparison.png", dpi=300)
print("\nPlot saved as 'seen_traj01_comparison.png'. Displaying window...")
plt.show()
