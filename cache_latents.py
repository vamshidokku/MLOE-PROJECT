import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Pre-extracting ResNet-18 latents using: {device}")

# Pre-trained frozen backbone
weights = ResNet18_Weights.DEFAULT
base_resnet = resnet18(weights=weights)
backbone = torch.nn.Sequential(*list(base_resnet.children())[:-1]).to(device)
backbone.eval()

for p in backbone.parameters():
    p.requires_grad = False

trajectories = ["traj_01", "traj_02", "traj_03", "traj_04", "traj_05", "traj_06"]
img_size = (224, 224)
batch_size = 64

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

for traj in trajectories:
    data_dir = Path("processed_data") / traj
    img_dir = data_dir / "cam0_images"
    csv_path = data_dir / "images.csv"
    out_file = data_dir / "resnet_latents.pt"

    if not csv_path.exists() or not img_dir.exists():
        continue

    print(f"\nProcessing {traj}...")
    df = pd.read_csv(csv_path)
    filenames = df["filename"].values
    num_imgs = len(filenames)

    all_features = []

    for i in tqdm(range(0, num_imgs, batch_size), desc=f"Extracting {traj}"):
        batch_files = filenames[i : i + batch_size]
        batch_tensors = []

        for fname in batch_files:
            img = cv2.imread(str(img_dir / fname))
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - mean) / std
            batch_tensors.append(torch.from_numpy(img).permute(2, 0, 1))

        batch_tensor = torch.stack(batch_tensors, dim=0).to(device)

        with torch.no_grad():
            feats = backbone(batch_tensor).squeeze(-1).squeeze(-1) # (B, 512)
            all_features.append(feats.cpu())

    all_features = torch.cat(all_features, dim=0) # (N, 512)
    torch.save(all_features, out_file)
    print(f"Saved {all_features.shape[0]} feature vectors ({out_file.stat().st_size / (1024*1024):.2f} MB) -> {out_file.name}")

print("\nAll trajectory latents cached successfully!")
