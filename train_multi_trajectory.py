import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet18, ResNet18_Weights
from dataset_vo import UnderwaterVODataset
from tqdm import tqdm
from pathlib import Path

# ----------------- DeepVO Architecture -----------------
class DeepVOModel(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepVOModel, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = resnet18(weights=weights)

        # 1x1 conv mapping 2 stacked grayscale frames (t, t+1) to 3 input channels
        self.input_conv = nn.Conv2d(2, 3, kernel_size=1, bias=False)
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-1])
        
        # 6-DoF Regressor head: [dx, dy, dz, droll, dpitch, dyaw]
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        x = self.input_conv(x)
        features = self.backbone(x)
        return self.regressor(features)

# ----------------- Relative Pose Loss -----------------
class PoseLoss(nn.Module):
    def __init__(self, beta=100.0):
        super(PoseLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.beta = beta  # Balance rotation radians vs translation meters

    def forward(self, pred, target):
        loss_pos = self.mse(pred[:, :3], target[:, :3])
        loss_rot = self.mse(pred[:, 3:], target[:, 3:])
        return loss_pos + self.beta * loss_rot, loss_pos, loss_rot

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing multi-trajectory training on: {device}")

    # 1. Load Training Trajectories (01 to 05)
    train_traj_names = ["traj_01", "traj_02", "traj_03", "traj_04", "traj_05"]
    train_datasets = []

    print("\nInitializing datasets for training pool...")
    for traj in train_traj_names:
        p = Path("processed_data") / traj
        if (p / "images.csv").exists() and list(p.glob("*mocap*.tum")):
            print(f"Loading {traj}...")
            train_datasets.append(UnderwaterVODataset(data_dir=str(p)))

    # Concatenate training sets
    combined_train_ds = ConcatDataset(train_datasets)
    print(f"--> Total combined training frame pairs: {len(combined_train_ds)}")

    # 2. Load Unseen Trajectory for Validation (traj_06)
    val_p = Path("processed_data") / "traj_06"
    print(f"Loading validation set ({val_p.name})...")
    val_ds = UnderwaterVODataset(data_dir=str(val_p))
    print(f"--> Total validation frame pairs: {len(val_ds)}")

    train_loader = DataLoader(combined_train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # 3. Model, Loss, Optimizer
    model = DeepVOModel(pretrained=True).to(device)
    criterion = PoseLoss(beta=100.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    epochs = 15
    best_val_loss = float('inf')

    print(f"\nBeginning training across {len(combined_train_ds)} pairs | Validating on {len(val_ds)} unseen pairs...\n")

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Train]"):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss, _, _ = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(combined_train_ds)

        # Validation phase on unseen traj_06
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Val]"):
                images, targets = images.to(device), targets.to(device)
                preds = model(images)
                loss, _, _ = criterion(preds, targets)
                running_val_loss += loss.item() * images.size(0)

        epoch_val_loss = running_val_loss / len(val_ds)
        scheduler.step(epoch_val_loss)

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.6f} | Val Loss (traj_06): {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_multi_traj_vo.pth")
            print("  --> Saved new best generalization checkpoint: best_multi_traj_vo.pth")

    print("\nMulti-trajectory training completed successfully!")

if __name__ == "__main__":
    main()
