import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from dataset_cached_lstm import CachedSequenceDataset
from pathlib import Path

class PureLSTMVO(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2):
        super(PureLSTMVO, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)             # (B, Seq_Len, hidden_dim)
        step_outputs = lstm_out[:, :-1, :]     # (B, Seq_Len - 1, hidden_dim)
        poses = self.fc(step_outputs)          # (B, Seq_Len - 1, 6)
        return poses

class SequencePoseLoss(nn.Module):
    def __init__(self, rot_weight=100.0):
        super(SequencePoseLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.rot_weight = rot_weight

    def forward(self, pred, target):
        loss_pos = self.mse(pred[:, :, :3], target[:, :, :3])
        loss_rot = self.mse(pred[:, :, 3:], target[:, :, 3:])
        return loss_pos + self.rot_weight * loss_rot

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Pure LSTM on cached latents: {device}")

    SEQ_LEN = 200
    STRIDE = 50

    train_trajs = ["traj_01", "traj_02", "traj_03", "traj_04", "traj_05"]
    train_datasets = [
        CachedSequenceDataset(f"processed_data/{t}", seq_len=SEQ_LEN, step_stride=STRIDE)
        for t in train_trajs
    ]
    train_ds = ConcatDataset(train_datasets)
    val_ds = CachedSequenceDataset("processed_data/traj_06", seq_len=SEQ_LEN, step_stride=100)

    print(f"Training windows: {len(train_ds)} | Validation windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = PureLSTMVO(input_dim=512, hidden_dim=256, num_layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)
    criterion = SequencePoseLoss(rot_weight=100.0)

    epochs = 40
    best_val_loss = float('inf')

    print("\nBeginning accelerated LSTM training...\n")

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for seq_feats, seq_targets in train_loader:
            seq_feats, seq_targets = seq_feats.to(device), seq_targets.to(device)

            optimizer.zero_grad()
            preds = model(seq_feats)
            loss = criterion(preds, seq_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * seq_feats.size(0)

        epoch_train_loss = running_train_loss / len(train_ds)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for seq_feats, seq_targets in val_loader:
                seq_feats, seq_targets = seq_feats.to(device), seq_targets.to(device)
                preds = model(seq_feats)
                loss = criterion(preds, seq_targets)
                running_val_loss += loss.item() * seq_feats.size(0)

        epoch_val_loss = running_val_loss / len(val_ds)
        scheduler.step(epoch_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch_val_loss < best_val_loss:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_pure_lstm_vo.pth")

    print("\nTraining complete! Saved weights to: best_pure_lstm_vo.pth")

if __name__ == "__main__":
    main()
