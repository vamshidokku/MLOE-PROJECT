import matplotlib.pyplot as plt
from evo.tools import file_interface
from evo.core import sync, metrics

def evaluate_and_plot(gt_file, est_file, traj_title, ax):
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
            label="Ground Truth (MoCap)", color="black", linestyle="--", linewidth=1.5)
    ax.plot(traj_est.positions_xyz[:, 0], traj_est.positions_xyz[:, 1], traj_est.positions_xyz[:, 2], 
            label="LSTM Predicted", color="dodgerblue", linewidth=1.5)
    ax.scatter(traj_ref.positions_xyz[0, 0], traj_ref.positions_xyz[0, 1], traj_ref.positions_xyz[0, 2],
               color="lime", s=80, edgecolors="k", label="Origin (t=0)")

    ax.set_title(f"{traj_title}\nATE RMSE: {stats['rmse']:.3f} m")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()

fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(121, projection='3d')
evaluate_and_plot("processed_data/traj_06/traj_06_mocap.tum", "lstm_estimated_traj06.tum", "Unseen traj_06", ax1)

ax2 = fig.add_subplot(122, projection='3d')
evaluate_and_plot("processed_data/traj_01/traj_01_mocap.tum", "lstm_estimated_traj01.tum", "Seen traj_01", ax2)

plt.tight_layout()
plt.savefig("lstm_evaluation_comparison.png", dpi=300)
print("\nSaved comparison figure to 'lstm_evaluation_comparison.png'. Displaying plots...")
plt.show()
