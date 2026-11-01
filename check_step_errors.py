import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset_vo import UnderwaterVODataset
import torch
from train_multi_trajectory import DeepVOModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = UnderwaterVODataset(data_dir="processed_data/traj_06")
model = DeepVOModel(pretrained=False).to(device)
model.load_state_dict(torch.load("best_multi_traj_vo.pth", map_location=device))
model.eval()

preds = []
targets = dataset.rel_poses

with torch.no_grad():
    for i in range(min(500, len(dataset))):
        img_pair, _ = dataset[i]
        p = model(img_pair.unsqueeze(0).to(device)).cpu().numpy()[0]
        preds.append(p)

preds = np.array(preds)
targets = targets[:len(preds)]

fig, axs = plt.subplots(2, 3, figsize=(14, 7))
labels = ['dx (m)', 'dy (m)', 'dz (m)', 'droll (rad)', 'dpitch (rad)', 'dyaw (rad)']

for idx, ax in enumerate(axs.flat):
    ax.plot(targets[:, idx], label="Ground Truth", color="black", alpha=0.7)
    ax.plot(preds[:, idx], label="CNN Predicted", color="red", alpha=0.7)
    ax.set_title(labels[idx])
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("step_error_breakdown.png", dpi=300)
print("Saved step-level breakdown to 'step_error_breakdown.png'. Displaying...")
plt.show()
