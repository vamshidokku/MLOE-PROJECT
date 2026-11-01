import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Precomputing dense optical flow using OpenCV on: {device}")

flow_encoder = nn.Sequential(
    nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((4, 4)),
    nn.Flatten(),
    nn.Linear(128 * 4 * 4, 512),
    nn.LayerNorm(512)
).to(device)

flow_encoder.eval()
for p in flow_encoder.parameters():
    p.requires_grad = False

trajectories = ["traj_01", "traj_02", "traj_03", "traj_04", "traj_05", "traj_06"]
flow_size = (224, 224)

for traj in trajectories:
    data_dir = Path("processed_data") / traj
    img_dir = data_dir / "cam0_images"
    csv_path = data_dir / "images.csv"
    out_file = data_dir / "flow_latents.pt"

    if not csv_path.exists():
        continue

    df = pd.read_csv(csv_path)
    filenames = df["filename"].values
    num_frames = len(filenames)

    print(f"\nComputing Farnebäck optical flow for {traj} ({num_frames} frames)...")

    prev_path = img_dir / filenames[0]
    prev_gray = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
    prev_gray = cv2.resize(prev_gray, flow_size)

    flow_vectors = []
    batch_tensors = []
    batch_size = 64

    for i in tqdm(range(1, num_frames), desc=f"Flow {traj}"):
        curr_path = img_dir / filenames[i]
        curr_gray = cv2.imread(str(curr_path), cv2.IMREAD_GRAYSCALE)
        curr_gray = cv2.resize(curr_gray, flow_size)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        prev_gray = curr_gray

        flow = np.clip(flow / 20.0, -1.0, 1.0)
        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()
        batch_tensors.append(flow_tensor)

        if len(batch_tensors) == batch_size or i == num_frames - 1:
            batch_stacked = torch.stack(batch_tensors, dim=0).to(device)
            with torch.no_grad():
                encoded = flow_encoder(batch_stacked)
                flow_vectors.append(encoded.cpu())
            batch_tensors = []

    flow_vectors = torch.cat(flow_vectors, dim=0)
    zero_pad = torch.zeros((1, 512))
    flow_vectors = torch.cat([zero_pad, flow_vectors], dim=0)

    torch.save(flow_vectors, out_file)
    print(f"Saved flow latents ({flow_vectors.shape[0]} steps) -> {out_file.name}")

print("\nDense optical flow extraction completed!")
