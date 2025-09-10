import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    """
    Core building block of WaveNet architecture for time series processing
    
    This block implements the essential components that make WaveNet effective:
    1. Dilated convolution for exponentially growing receptive field
    2. Gated activation mechanism (key innovation from WaveNet)
    3. Residual and skip connections for training stability and multi-scale features
    4. Time embedding injection for diffusion timestep conditioning
    
    The combination of these elements allows the network to capture both
    local patterns and long-range dependencies in time series data.
    """
    def __init__(self, residual_channels, skip_channels, time_emb_dim, dilation):
        """
        Initialize residual block components
        
        Args:
            residual_channels: Number of channels in residual path (internal hidden dimension)
            skip_channels: Number of channels in skip connections (for multi-scale fusion)
            time_emb_dim: Dimension of time embedding for diffusion conditioning
            dilation: Dilation factor for convolution (1, 2, 4, 8, 16, 32, ...)
        """
        super().__init__()
        
        # Dilated convolution: the heart of WaveNet's temporal modeling capability
        # Dilation allows exponential growth of receptive field without linear cost increase
        # Each layer sees 2^k times more context than standard convolution
        self.dilated_conv = nn.Conv1d(
            residual_channels,          # Input channels from previous layer
            2 * residual_channels,      # Output 2x channels for gated activation split
            3,                          # Kernel size of 3 (good balance of context and efficiency)
            padding=dilation,           # Padding equals dilation for 'same' output size
            dilation=dilation,          # Dilation factor (1, 2, 4, 8, 16, 32, ...)
            padding_mode="circular",    # Circular padding appropriate for time series
        )
        
        # Time embedding projection: incorporates diffusion timestep information
        # This allows the same network to behave differently at different noise levels
        # Critical for diffusion models where behavior must adapt to noise amount
        self.time_mlp = nn.Linear(time_emb_dim, residual_channels)
        
        # Output projection: splits processing into residual and skip paths
        # Residual path: continues to next layer (local processing)
        # Skip path: goes directly to output (multi-scale feature collection)
        self.output_projection = nn.Conv1d(residual_channels, residual_channels + skip_channels, 1)

        # Weight initialization: Kaiming normal good for ReLU-like activations
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, time_emb):
        """
        Forward pass through residual block
        
        Args:
            x: Input tensor [batch, residual_channels, time]
            time_emb: Time embedding [batch, time_emb_dim] - diffusion timestep info
            
        Returns:
            tuple: (residual_output, skip_output) for next layer and skip connection
        """
        # Process time embedding and add time dimension for broadcasting
        # Shape transformation: [batch, time_emb_dim] → [batch, residual_channels] → [batch, residual_channels, 1]
        time_emb = self.time_mlp(time_emb).unsqueeze(-1)  # Add time dimension

        # Add time embedding to input (broadcast across time dimension)
        # This conditions the processing on the current diffusion timestep
        # Different timesteps (noise levels) require different processing strategies
        y = x + time_emb

        # Apply dilated convolution
        # Output has 2x channels to support gated activation mechanism
        y = self.dilated_conv(y)

        # Gated activation: the key innovation that makes WaveNet work
        # Split output into two equal parts: gate and filter
        gate, filter_gate = torch.chunk(y, 2, dim=1)
        
        # Apply gated activation: σ(gate) ⊙ tanh(filter)
        # σ(gate): learns what information to keep/forget (like LSTM forget gate)
        # tanh(filter): learns what new information to add (like LSTM input gate)
        # This allows dynamic, content-dependent information flow control
        y = torch.sigmoid(gate) * torch.tanh(filter_gate)

        # Output projection: transform to residual + skip dimensions
        y = self.output_projection(y)

        # Split into residual and skip connections
        # residual: continues to next layer (maintains gradient flow)
        # skip: goes directly to output (captures features at this temporal scale)
        residual, skip = torch.split(y, [x.shape[1], y.shape[1] - x.shape[1]], dim=1)

        # Apply residual connection with normalization
        # Division by sqrt(2) maintains variance across layers (similar to ResNet)
        # This prevents activation magnitude from growing/shrinking during training
        return (x + residual) / math.sqrt(2.0), skip

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional embeddings for encoding diffusion timesteps
    
    This creates unique, continuous embeddings for each timestep that allow
    the model to understand its position in the diffusion process.
    
    Based on the positional encoding from "Attention Is All You Need",
    adapted for diffusion timestep encoding. Different frequencies ensure
    each timestep gets a unique, learnable representation.
    """
    def __init__(self, dim):
        """
        Initialize sinusoidal embedding layer
        
        Args:
            dim: Dimension of the embedding (should be even for sin/cos pairs)
        """
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Generate sinusoidal embeddings for given timesteps
        
        Creates embeddings using multiple frequencies:
        - High frequencies: change rapidly between adjacent timesteps
        - Low frequencies: change slowly, capture coarse temporal structure
        
        Args:
            time: Timestep indices [batch_size] - which diffusion step
            
        Returns:
            torch.Tensor: Sinusoidal embeddings [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        
        # Create frequency components for sinusoidal encoding
        # Higher frequencies change faster, lower frequencies change slower
        # This provides a rich, unique encoding for each timestep
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Apply frequencies to timesteps
        # Broadcasting: time[:, None] expands timesteps, embeddings[None, :] expands frequencies
        embeddings = time[:, None] * embeddings[None, :]
        
        # Concatenate sin and cos components for full embedding
        # This doubles the dimension and ensures each timestep gets unique representation
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class WaveNetDiffusion(nn.Module):
    """
    WaveNet-style diffusion model for time series data
    
    This architecture replaces traditional U-Net designs with a WaveNet-based approach
    specifically optimized for sequential/temporal data like financial time series.
    
    Key innovations over standard approaches:
    1. Dilated convolutions: exponentially growing receptive field without quadratic cost
    2. Gated activations: dynamic information flow control
    3. Skip connections: multi-scale feature fusion from all temporal resolutions
    4. Circular padding: appropriate boundary handling for time series
    5. Time conditioning: diffusion timestep awareness throughout the network
    
    This design is particularly well-suited for financial time series because:
    - Captures both short-term (daily) and long-term (monthly) patterns
    - Handles variable-length dependencies efficiently
    - Preserves temporal structure better than spatial convolutions
    """
    def __init__(self, input_dim=4, residual_channels=64, skip_channels=64, num_layers=10, time_emb_dim=32):
        """
        Initialize WaveNet diffusion model
        
        Args:
            input_dim: Number of input features (e.g., 4 for OHLC data)
            residual_channels: Hidden dimension for residual connections (model capacity)
            skip_channels: Hidden dimension for skip connections (output features)
            num_layers: Number of residual blocks (more layers = larger receptive field)
            time_emb_dim: Dimension of time embeddings (timestep conditioning)
        """
        super().__init__()

        # Time embedding network: converts timestep indices to dense representations
        # This allows the model to understand which stage of diffusion it's processing
        # Different noise levels require different denoising strategies
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),  # Convert timestep to sinusoidal encoding
            nn.Linear(time_emb_dim, time_emb_dim),       # Learn from the encoding
            nn.ReLU()                                    # Non-linearity for learning
        )

        # Initial projection: map input features to internal hidden representation
        # Transforms from input space (e.g., 4 OHLC features) to hidden space (e.g., 64 channels)
        self.input_projection = nn.Conv1d(input_dim, residual_channels, 1)

        # Stack of residual blocks with carefully designed dilation pattern
        # Dilation pattern: 1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32, ...
        # This creates exponentially growing receptive field that resets periodically
        # Prevents dilation from becoming too large (which can hurt performance)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                residual_channels=residual_channels,
                skip_channels=skip_channels,
                time_emb_dim=time_emb_dim,
                dilation=2 ** (i % 6)  # Cycle through 1, 2, 4, 8, 16, 32
            )
            for i in range(num_layers)
        ])

        # Output projection: transform accumulated skip connections back to input space
        # Skip connections contain multi-scale features from all temporal resolutions
        # Final projection combines these into noise prediction
        self.output_projection = nn.Sequential(
            nn.ReLU(),                                    # Non-linearity
            nn.Conv1d(skip_channels, skip_channels, 1),   # Intermediate feature processing
            nn.ReLU(),                                    # Another non-linearity
            nn.Conv1d(skip_channels, input_dim, 1)        # Final projection to input dimension
        )

    def forward(self, x, timestep):
        """
        Forward pass through WaveNet diffusion model
        
        The model processes noisy input and predicts the noise that was added,
        conditioned on the diffusion timestep.
        
        Args:
            x: [batch_size, input_dim, sequence_length] - noisy time series input
            timestep: [batch_size] - diffusion timestep for each sample in batch
            
        Returns:
            [batch_size, input_dim, sequence_length] - predicted noise (same shape as input)
        """
        # Generate time embeddings for conditioning
        # Converts integer timesteps to rich representations the network can use
        t = self.time_mlp(timestep)

        # Project input to internal hidden representation
        # Transform from input space to hidden space for internal processing
        x = self.input_projection(x)

        # Pass through residual blocks and collect skip connections
        # Each block processes information at a different temporal scale
        # Skip connections capture important patterns at each scale
        skip_connections = []
        for block in self.residual_blocks:
            x, skip = block(x, t)  # x continues to next layer, skip saved for output
            skip_connections.append(skip)

        # Combine all skip connections (multi-scale feature fusion)
        # Each skip connection captures patterns at different temporal scales:
        # - Early layers: high-frequency patterns (daily fluctuations)
        # - Later layers: low-frequency patterns (long-term trends)
        # Summing combines information across all temporal scales
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)

        # Generate final output: predicted noise
        # Transform multi-scale features back to original input space
        # Output represents the noise that should be removed at this timestep
        return self.output_projection(skip_sum)

# PRESERVED: Your exact factory function and comments
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

# ============================================================================
# ARCHITECTURE DESIGN RATIONALE
# ============================================================================
"""
WHY WAVENET FOR TIME SERIES DIFFUSION?

1. TEMPORAL STRUCTURE AWARENESS:
   - Time series are inherently sequential, unlike images (2D spatial)
   - WaveNet was designed specifically for sequential data (originally audio)
   - Dilated convolutions naturally respect temporal ordering

2. EFFICIENT LONG-RANGE DEPENDENCIES:
   - Receptive field grows exponentially: O(2^k) context with k layers
   - Standard RNNs: sequential processing, hard to parallelize
   - Standard CNNs: linear growth, requires many layers for long context
   - WaveNet: exponential growth with parallel computation

3. MULTI-SCALE FEATURE EXTRACTION:
   - Skip connections capture patterns at multiple temporal scales
   - Early layers: high-frequency patterns (intraday movements)
   - Middle layers: medium-frequency patterns (weekly cycles)  
   - Later layers: low-frequency patterns (monthly/seasonal trends)
   - All scales contribute to final prediction

4. GATED ACTIVATION BENEFITS:
   - σ(gate) ⊙ tanh(filter) provides content-dependent information flow
   - Unlike fixed activations (ReLU), gates can selectively forget/remember
   - Critical for complex temporal patterns in financial data
   - Helps model learn what information is relevant at each timestep

5. DIFFUSION-SPECIFIC ADVANTAGES:
   - Time embedding injection at each layer
   - Model behavior adapts to noise level (early vs late diffusion steps)
   - Circular padding appropriate for cyclical time series patterns
   - Stable training due to residual connections

RECEPTIVE FIELD CALCULATION:
With dilation pattern [1,2,4,8,16,32] and 10 layers:
- Layer 1: sees 3 timesteps (dilation=1, kernel=3)
- Layer 2: sees 3+2*2 = 7 timesteps (dilation=2)
- Layer 3: sees 7+2*4 = 15 timesteps (dilation=4)
- Layer 4: sees 15+2*8 = 31 timesteps (dilation=8)
- Layer 5: sees 31+2*16 = 63 timesteps (dilation=16)
- Layer 6: sees 63+2*32 = 127 timesteps (dilation=32)

Total effective receptive field: 127+ timesteps
This captures both short-term and long-term dependencies efficiently.

COMPARISON TO ALTERNATIVES:

U-Net:
+ Good for spatial data with clear encoder-decoder structure
- Designed for 2D images, not 1D sequences
- Downsampling/upsampling may lose temporal precision
- Skip connections connect spatially distant features, not temporally

Transformer:
+ Excellent for very long sequences
+ Self-attention captures all pairwise dependencies
- Quadratic computational cost O(n²)
- Less inductive bias for local temporal structure
- May overfit with limited financial data

RNN/LSTM:
+ Natural for sequential data
+ Can handle variable-length sequences
- Sequential computation, hard to parallelize
- Vanishing gradients for very long sequences
- Slower training than convolution-based models

WaveNet strikes the optimal balance for time series diffusion:
- Temporal structure awareness
- Efficient parallel computation
- Multi-scale feature extraction
- Stable training dynamics
"""

# PRESERVED: Your exact example usage documentation
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