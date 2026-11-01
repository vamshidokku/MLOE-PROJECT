import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from dataset_lstm_vo import UnderwaterSequenceVODataset
from model_lstm_vo import CNNLSTMVO
from tqdm import tqdm
from pathlib import Path

class SequencePoseLoss(nn.Module):
    def __init__(self, rot_weight=100.0):
        super(SequencePoseLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.rot_weight = rot_weight

    def forward(self, pred, target):
        loss_pos = self.mse(pred[:, :, :3], target[:, :, :3])
        loss_rot = self.mse(pred[:, :, 3:], target[:, :, 3:])
        return loss_pos + self.rot_weight * loss_rot, loss_pos, loss_rot

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Sequence CNN-LSTM Training on: {device}")

    # Window configuration
    SEQ_LEN = 200
    STRIDE = 100

    train_trajs = ["traj_01", "traj_02", "traj_03", "traj_04", "traj_05"]
    train_datasets = []

    print("\nExtracting 200-frame sequence windows across training trajectories...")
    for traj in train_trajs:
        p = Path("processed_data") / traj
        if (p / "images.csv").exists():
            ds = UnderwaterSequenceVODataset(data_dir=str(p), seq_len=SEQ_LEN, step_stride=STRIDE)
            print(f"Loaded {traj}: {len(ds)} sequence windows")
            train_datasets.append(ds)

    train_ds = ConcatDataset(train_datasets)
    print(f"--> Total training sequence windows: {len(train_ds)}")

    val_ds = UnderwaterSequenceVODataset(
        data_dir="processed_data/traj_06", 
        seq_len=SEQ_LEN, 
        step_stride=200
    )
    print(f"--> Total validation sequence windows (traj_06): {len(val_ds)}\n")

    # Use batch_size = 4 (each item is 200 frames, so 4 * 200 = 800 frames per forward step)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    # Initialize model with frozen CNN
    model = CNNLSTMVO(hidden_dim=256, num_layers=2, freeze_cnn=True).to(device)
    
    # Train only LSTM and regression head
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = SequencePoseLoss(rot_weight=100.0)

    epochs = 25
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for seq_imgs, seq_targets in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Train]"):
            seq_imgs = seq_imgs.to(device)
            seq_targets = seq_targets.to(device)

            optimizer.zero_grad()
            preds = model(seq_imgs)
            loss, _, _ = criterion(preds, seq_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * seq_imgs.size(0)

        epoch_train_loss = running_train_loss / len(train_ds)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for seq_imgs, seq_targets in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Val]"):
                seq_imgs = seq_imgs.to(device)
                seq_targets = seq_targets.to(device)
                preds = model(seq_imgs)
                loss, _, _ = criterion(preds, seq_targets)
                running_val_loss += loss.item() * seq_imgs.size(0)

        epoch_val_loss = running_val_loss / len(val_ds)
        scheduler.step(epoch_val_loss)

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_cnn_lstm_vo.pth")
            print("  --> Saved new best checkpoint: best_cnn_lstm_vo.pth")

    print("\nCNN-LSTM training completed successfully!")

if __name__ == "__main__":
    main()
