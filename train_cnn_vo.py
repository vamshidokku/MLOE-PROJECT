import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
from dataset_vo import UnderwaterVODataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------- Model Architecture -----------------
class DeepVOModel(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepVOModel, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = resnet18(weights=weights)

        # 1x1 conv to map 2-channel input (Frame_t, Frame_t+1) to 3 channels for ResNet
        self.input_conv = nn.Conv2d(2, 3, kernel_size=1, bias=False)
        
        # Backbone: all layers except final fully connected layer
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-1])
        
        # 6-DoF pose regressor: [dx, dy, dz, droll, dpitch, dyaw]
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
        pose_6dof = self.regressor(features)
        return pose_6dof

# ----------------- Custom Pose Loss -----------------
class PoseLoss(nn.Module):
    def __init__(self, beta=100.0):
        super(PoseLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.beta = beta  # Weight factor for angular rotation loss

    def forward(self, pred, target):
        loss_pos = self.mse(pred[:, :3], target[:, :3])
        loss_rot = self.mse(pred[:, 3:], target[:, 3:])
        return loss_pos + self.beta * loss_rot, loss_pos, loss_rot

# ----------------- Training Loop -----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset and create train/val split
    full_dataset = UnderwaterVODataset()
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = DeepVOModel(pretrained=True).to(device)
    criterion = PoseLoss(beta=100.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    epochs = 15
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    print(f"\nStarting training on {train_size} training pairs, validating on {val_size} pairs...")

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss, l_pos, l_rot = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / train_size
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, targets = images.to(device), targets.to(device)
                preds = model(images)
                loss, _, _ = criterion(preds, targets)
                running_val_loss += loss.item() * images.size(0)

        epoch_val_loss = running_val_loss / val_size
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        # Save checkpoint if best
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_cnn_vo.pth")
            print(f"  --> Saved new best checkpoint (best_cnn_vo.pth)")

    print("\nTraining complete! Checkpoint saved as best_cnn_vo.pth")

if __name__ == "__main__":
    main()
