"""Hybrid CNN model with residual blocks and Squeeze-and-Excitation attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = channels // reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.view(b, c, 1, 1)


class PreActResBlock(nn.Module):
    """Pre-activation residual block with SE attention and optional stochastic depth."""

    def __init__(self, channels, se_reduction=4, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.drop = nn.Dropout2d(dropout)
        self.se = SEBlock(channels, se_reduction)
        self.drop_path = drop_path

    def forward(self, x):
        identity = x
        # Stochastic depth: skip entire block with probability drop_path during training
        if self.training and self.drop_path > 0.0 and torch.rand(1).item() < self.drop_path:
            return identity
        out = F.silu(self.bn1(x))
        out = self.conv1(out)
        out = F.silu(self.bn2(out))
        out = self.conv2(out)
        out = self.drop(out)
        out = self.se(out)
        # Scale output at inference to compensate for stochastic depth
        if not self.training and self.drop_path > 0.0:
            out = out * (1.0 - self.drop_path)
        return out + identity


class ChessCNNModel(nn.Module):
    """
    Hybrid CNN for chess position complexity prediction.

    Input: (batch, 18, 8, 8) tensor from fen_to_tensor()
      - Channels 0-11: piece planes (12 piece types)
      - Channel 12: side-to-move (1=white, 0=black)
      - Channels 13-16: castling rights (K, Q, k, q)
      - Channel 17: en passant square

    Output: (batch, 1) positive scalar — predicted accuracy loss (win% reduction).
    """

    def __init__(self, channels=128, num_blocks=6, se_reduction=4,
                 block_dropout=0.1, head_dropout=0.3, stochastic_depth=0.0):
        super().__init__()

        # Stem: 12 piece planes → channels
        self.stem_conv = nn.Conv2d(12, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)

        # Tower: residual blocks (with linearly increasing drop path)
        self.tower = nn.ModuleList([
            PreActResBlock(channels, se_reduction, block_dropout,
                          drop_path=stochastic_depth * (i / max(1, num_blocks - 1)))
            for i in range(num_blocks)
        ])

        # Head
        self.head_bn = nn.BatchNorm2d(channels)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Metadata: 1 (side-to-move) + 4 (castling) + 7 (EP files a-g) = 12
        meta_size = 12
        head_input = channels + meta_size  # 128 + 12 = 140

        self.head_mlp = nn.Sequential(
            nn.Linear(head_input, 256),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Split input: board planes vs metadata
        board = x[:, 0:12, :, :]                       # (B, 12, 8, 8)
        side_to_move = x[:, 12, 0, 0].unsqueeze(1)     # (B, 1)
        castling = x[:, 13:17, 0, 0]                    # (B, 4)
        ep_files = x[:, 17, :, :].sum(dim=1)[:, :7]     # (B, 7)
        metadata = torch.cat([side_to_move, castling, ep_files], dim=1)  # (B, 12)

        # Stem
        h = F.silu(self.stem_bn(self.stem_conv(board)))

        # Tower
        for block in self.tower:
            h = block(h)

        # Head
        h = F.silu(self.head_bn(h))
        h = self.gap(h).flatten(1)                       # (B, 128)
        h = torch.cat([h, metadata], dim=1)              # (B, 140)
        h = self.head_mlp(h)                             # (B, 1)

        return h
