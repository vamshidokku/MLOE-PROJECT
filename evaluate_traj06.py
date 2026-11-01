import matplotlib.pyplot as plt
from evo.tools import file_interface
from evo.core import sync, metrics

gt_file = "processed_data/traj_06/traj_06_mocap.tum"
est_file = "multi_traj_estimated_traj06.tum"

print("Loading trajectories...")
traj_ref = file_interface.read_tum_trajectory_file(gt_file)
traj_est = file_interface.read_tum_trajectory_file(est_file)

traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.05)
traj_est.align(traj_ref, correct_scale=False)

ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
ape_metric.process_data((traj_ref, traj_est))
ape_stats = ape_metric.get_all_statistics()

print("\n" + "="*45)
print("     UNSEEN TRAJ_06 EVALUATION RESULTS")
print("="*45)
for k, v in ape_stats.items():
    print(f"{k.upper():<10} : {v:.4f} m")
print("="*45)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(traj_ref.positions_xyz[:, 0], traj_ref.positions_xyz[:, 1], traj_ref.positions_xyz[:, 2], 
        label="MoCap Ground Truth (traj_06)", color="black", linestyle="--", linewidth=1.5)
ax.plot(traj_est.positions_xyz[:, 0], traj_est.positions_xyz[:, 1], traj_est.positions_xyz[:, 2], 
        label="Multi-Traj DeepVO", color="crimson", linewidth=1.5)

ax.set_title(f"Unseen Trajectory 06: Ground Truth vs Multi-Traj CNN\nATE RMSE: {ape_stats['rmse']:.3f} m")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.legend()
plt.tight_layout()
plt.savefig("multi_traj06_comparison.png", dpi=300)
print("\nSaved comparison plot to 'multi_traj06_comparison.png'. Displaying plot window...")
plt.show()
