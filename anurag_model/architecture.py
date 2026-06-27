import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.2):
        """
        An upgraded 1D residual block using dilated convolutions to expand the 
        receptive field without losing sequence resolution.
        """
        super(DilatedResidualBlock1D, self).__init__()
        
        # First convolution: standard kernel, handles downsampling if stride > 1
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=5, stride=stride, padding=2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution: Dilated convolution to capture wider temporal structures
        # To maintain length: padding = 2 * dilation (since kernel_size = 5, stride = 1)
        padding = 2 * dilation
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, 
            kernel_size=5, stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class MultiHeadSelfAttention1D(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        """
        An upgraded Multi-Head Self-Attention layer for 1D temporal sequences.
        Splits channels into separate heads to capture multiple different periodicities
        and stellar patterns simultaneously.
        """
        super(MultiHeadSelfAttention1D, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"
        
        # Query, Key, and Value projections using 1x1 convolutions
        self.q_proj = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.v_proj = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
        # Final output mixing projection
        self.out_proj = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
        # Learnable gating parameter initialized to 0
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Input shape: [Batch, Channels, SeqLen]
        batch_size, channels, seq_len = x.size()
        
        # 1. Project Query, Key, Value
        q = self.q_proj(x) # [B, C, L]
        k = self.k_proj(x) # [B, C, L]
        v = self.v_proj(x) # [B, C, L]
        
        # 2. Reshape and permute for multi-head calculation
        # [B, C, L] -> [B, H, D, L] -> transpose to [B, H, L, D]
        q = q.view(batch_size, self.num_heads, self.head_dim, seq_len).transpose(-1, -2)
        k = k.view(batch_size, self.num_heads, self.head_dim, seq_len) # [B, H, D, L]
        v = v.view(batch_size, self.num_heads, self.head_dim, seq_len).transpose(-1, -2) # [B, H, L, D]
        
        # 3. Calculate Scaled Dot-Product Attention
        # energy shape: [B, H, L, L]
        energy = torch.matmul(q, k) / (self.head_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)
        
        # 4. Multiply attention weights with Value
        # out shape: [B, H, L, D]
        out = torch.matmul(attention, v)
        
        # 5. Concatenate heads and project output
        # [B, H, L, D] -> transpose to [B, H, D, L] -> view as [B, C, L]
        out = out.transpose(-1, -2).contiguous().view(batch_size, channels, seq_len)
        out = self.out_proj(out)
        
        # 6. Apply residual connection gated by gamma
        out = self.gamma * out + x
        
        # Average attention maps across heads for Grad-CAM/visualization -> [B, L, L]
        mean_attention = torch.mean(attention, dim=1)
        
        return out, mean_attention

class LocalFeatureExtractor1D(nn.Module):
    def __init__(self, dropout=0.2):
        """
        Extracts high-resolution features from the zoomed-in local folded transit view.
        Since sequence length is small (200 points), a compact CNN layout is used.
        """
        super(LocalFeatureExtractor1D, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(dropout)
        
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x shape: [B, 1, 200]
        x = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        x = self.drop2(self.relu2(self.bn2(self.conv2(x))))
        x = self.drop3(self.relu3(self.bn3(self.conv3(x))))
        x = self.gap(x).squeeze(-1) # [B, 128]
        return x

class UpgradedExoplanetDetectorNet(nn.Module):
    def __init__(self, input_len=2000, dropout=0.3, num_heads=4):
        """
        The SOTA Upgraded Exoplanet Classification Network (Phase 2 Multi-Input).
        Integrates:
        - **Global View Branch (2000 pts)**: Processes the folded full orbit through
          Dilated Convolutions and Multi-Head Self-Attention.
        - **Local View Branch (200 pts)**: Processes the zoomed-in primary transit
          dip shape to check ingress/egress transit geometry.
        - **Feature Fusion**: Concatenates both representations for final classification.
        """
        super(UpgradedExoplanetDetectorNet, self).__init__()
        self.input_len = input_len
        
        # --- GLOBAL BRANCH SETUP ---
        self.global_conv = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.global_bn = nn.BatchNorm1d(32)
        self.global_relu = nn.ReLU(inplace=True)
        self.global_maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.global_res1 = DilatedResidualBlock1D(32, 64, stride=2, dilation=1, dropout=dropout)
        self.global_res2 = DilatedResidualBlock1D(64, 128, stride=2, dilation=2, dropout=dropout)
        self.global_res3 = DilatedResidualBlock1D(128, 256, stride=2, dilation=4, dropout=dropout)
        
        self.global_attention = MultiHeadSelfAttention1D(256, num_heads=num_heads)
        self.global_gap = nn.AdaptiveAvgPool1d(1)
        
        # --- LOCAL BRANCH SETUP ---
        self.local_branch = LocalFeatureExtractor1D(dropout=dropout)
        
        # --- CLASSIFICATION HEAD ---
        # Combines Global (256 features) and Local (128 features) branches
        self.fc1 = nn.Linear(256 + 128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, global_x, local_x):
        # Input shape: [Batch, SeqLen] or [Batch, 1, SeqLen]
        if len(global_x.shape) == 2:
            global_x = global_x.unsqueeze(1)
        if len(local_x.shape) == 2:
            local_x = local_x.unsqueeze(1)
            
        # 1. Global View Branch
        g = self.global_relu(self.global_bn(self.global_conv(global_x)))
        g = self.global_maxpool(g)
        g = self.global_res1(g)
        g = self.global_res2(g)
        g = self.global_res3(g)
        g, attn_map = self.global_attention(g)
        global_feats = self.global_gap(g).squeeze(-1) # [Batch, 256]
        
        # 2. Local View Branch
        local_feats = self.local_branch(local_x) # [Batch, 128]
        
        # 3. Feature Fusion & Classification
        fused = torch.cat([global_feats, local_feats], dim=1) # [Batch, 384]
        
        x = F.relu(self.fc1(fused))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x, attn_map
