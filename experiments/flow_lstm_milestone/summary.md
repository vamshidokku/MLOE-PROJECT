# Visual Odometry Milestone Comparison

| Model Architecture | Unseen (traj_06) ATE RMSE | Seen (traj_01) ATE RMSE | Qualitative Behavior |
| :--- | :--- | :--- | :--- |
| **Baseline 2-Frame CNN** | 36.39 m | 20.36 m | Mean-prediction collapse; straight line drift |
| **ResNet-18 + LSTM (200-frame window)** | 17.49 m | 22.26 m | Trajectory expansion; orientation divergence |
| **Flow + ResNet + LSTM** | 16.28 m | 9.88 m | Recovers angular turning and loop closure on traj_01 |
