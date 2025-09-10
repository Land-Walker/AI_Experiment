import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, time_emb_dim, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.time_mlp = nn.Linear(time_emb_dim, residual_channels)
        self.output_projection = nn.Conv1d(residual_channels, residual_channels + skip_channels, 1)

        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, time_emb):
        # Time embedding
        time_emb = self.time_mlp(time_emb).unsqueeze(-1)  # [batch, channels, 1]

        # Add time embedding to input
        y = x + time_emb

        # Dilated convolution
        y = self.dilated_conv(y)

        # Gated activation
        gate, filter_gate = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_gate)

        # Output projection
        y = self.output_projection(y)

        # Split into residual and skip connections
        residual, skip = torch.split(y, [x.shape[1], y.shape[1] - x.shape[1]], dim=1)

        return (x + residual) / math.sqrt(2.0), skip

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class WaveNetDiffusion(nn.Module):
    """
    WaveNet-style diffusion model for time series data.
    Replaces SimpleUnet for 1D time series.
    """
    def __init__(self, input_dim=4, residual_channels=64, skip_channels=64, num_layers=10, time_emb_dim=32):
        super().__init__()

        # Time embedding (same as U-Net)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection from input_dim to residual_channels
        self.input_projection = nn.Conv1d(input_dim, residual_channels, 1)

        # Stack of residual blocks with carefully controlled dilation
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                residual_channels=residual_channels,
                skip_channels=skip_channels,
                time_emb_dim=time_emb_dim,
                dilation=2 ** (i % 6)  # Cycle through 1, 2, 4, 8, 16, 32
            )
            for i in range(num_layers)
        ])

        # Output projection (back to input_dim)
        self.output_projection = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, input_dim, 1)
        )

    def forward(self, x, timestep):
        """
        Args:
            x: [batch_size, input_dim, sequence_length] - noisy time series
            timestep: [batch_size] - diffusion timestep
        Returns:
            [batch_size, input_dim, sequence_length] - predicted noise
        """
        # Time embedding
        t = self.time_mlp(timestep)

        # Initial projection
        x = self.input_projection(x)

        # Pass through residual blocks and collect skip connections
        skip_connections = []
        for block in self.residual_blocks:
            x, skip = block(x, t)
            skip_connections.append(skip)

        # Sum all skip connections
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)

        # Output projection
        return self.output_projection(skip_sum)

# Example usage - direct replacement for SimpleUnet
def create_wavenet_model(input_dim):
    """
    Factory function to create WaveNet model.
    Use this instead of SimpleUnet() for time series.
    """
    return WaveNetDiffusion(
        input_dim=input_dim,      # Number of features (e.g., 4 for OHLC)
        residual_channels=64,     # Internal feature dimension
        skip_channels=64,         # Skip connection dimension
        num_layers=10,            # Number of residual blocks
        time_emb_dim=32          # Time embedding dimension
    )


"""
Example Usage:

# For time series with 4 features (OHLC) and sequence length 100
model = create_wavenet_model(input_dim=4)

# Test forward pass
batch_size = 8
sequence_length = 100
x = torch.randn(batch_size, 4, sequence_length)  # [batch, features, time]
timestep = torch.randint(0, 1000, (batch_size,))

output = model(x, timestep)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Num params: {sum(p.numel() for p in model.parameters()):,}")

# Verify shapes match (essential for diffusion)
assert output.shape == x.shape, "Output shape must match input shape for diffusion!"
print("✅ Shape verification passed!")
"""