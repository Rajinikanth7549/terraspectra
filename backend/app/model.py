import torch
import torch.nn as nn

class TerraSpectraHybridNet(nn.Module):
    """
    Hybrid 3D-CNN + Vision Transformer (ViT) for Hyperspectral Crop Disease Detection.
    Input shape: (Batch, 1, 30, 11, 11) -> (Batch, Channels, Spectral_PCA, Height, Width)
    """
    def __init__(self, num_classes=4, pca_bands=30, patch_size=11, embed_dim=64):
        super().__init__()

        # 1. 3D-CNN Backbone (extracts local spatial + spectral features simultaneously)
        self.conv3d_block = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        # After two 3x3x3 convolutions:
        # Spectral depth: 30 -> 28 -> 26
        # Spatial size:   11 -> 9  -> 7
        flattened_spectral_channels = 16 * (pca_bands - 4)  # 16 * 26 = 416

        # 2. 2D Spatial-Spectral Projection into Token Embeddings
        self.conv2d_proj = nn.Sequential(
            nn.Conv2d(in_channels=flattened_spectral_channels, out_channels=embed_dim, kernel_size=3),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

        # Spatial size after 3x3 Conv2d: 7 -> 5 (5 x 5 = 25 tokens)
        num_tokens = (patch_size - 6) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, embed_dim) * 0.02)

        # 3. Vision Transformer (ViT) Self-Attention Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, 30, 11, 11)
        x = self.conv3d_block(x)  # (B, 16, 26, 7, 7)

        b, c, d, h, w = x.shape
        x = x.reshape(b, c * d, h, w)  # (B, 416, 7, 7)
        x = self.conv2d_proj(x)        # (B, 64, 5, 5)

        # Flatten spatial grid into sequence of tokens: (B, 25, 64)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embedding

        # Apply self-attention across tokens
        x = self.transformer(x)        # (B, 25, 64)

        # Global average pooling over tokens -> class logits
        x = x.mean(dim=1)              # (B, 64)
        logits = self.classifier(x)    # (B, 4)
        return logits