import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CNNLSTMVO(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2, freeze_cnn=True):
        super(CNNLSTMVO, self).__init__()

        # 1. Feature Extractor Backbone (Pre-trained ResNet-18)
        weights = ResNet18_Weights.DEFAULT
        base_resnet = resnet18(weights=weights)
        
        # Strip final classification FC layer -> outputs (Batch, 512, 1, 1)
        self.feature_extractor = nn.Sequential(*list(base_resnet.children())[:-1])

        if freeze_cnn:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # 2. Recurrent Sequential Processor (LSTM)
        # Input size is 512 (ResNet latent vector)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # 3. 6-DoF Motion Head: maps transition hidden states to relative motion
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, 3, H, W)
        batch_size, seq_len, C, H, W = x.shape

        # Flatten batch and sequence dimensions for efficient CNN forward pass
        x_flat = x.view(batch_size * seq_len, C, H, W)
        
        # Extract 512-dim features
        features = self.feature_extractor(x_flat)  # (B * L, 512, 1, 1)
        features = features.view(batch_size, seq_len, 512)

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(features)  # (B, Seq_Len, hidden_dim)

        # Each relative step t -> t+1 is predicted from the transition output
        # Take the first seq_len - 1 outputs to match the (seq_len - 1) pose transitions
        step_outputs = lstm_out[:, :-1, :]  # (B, Seq_Len - 1, hidden_dim)

        # Predict 6-DoF pose increments
        poses = self.fc(step_outputs)  # (B, Seq_Len - 1, 6)

        return poses
