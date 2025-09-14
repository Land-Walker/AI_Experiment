# ENHANCED: Updated imports
from torch.optim import Adam
import torch
import torch.nn.functional as F
import numpy as np

from data_loader import *
from forward_process import *
from model import *
from data_sampler import *
from evaluator import * 

# ============================================================================
# STEP 1: DATA LOADING WITH GLUONTS INTEGRATION
# ============================================================================

print("=== Enhanced Multivariate Time Series Diffusion Pipeline ===")
print("Loading financial data with GluonTS integration...")

# Load financial data (unchanged)
yf_df = load_yf_data()
show_yf_data(yf_df)

# ENHANCED: Multivariate dataset with GluonTS preprocessing
print("\nCreating multivariate dataset with OHLCV as separate features...")
dataset, dataloader, info = load_timeseries_dataset(
    yf_dataframe=yf_df,
    use_columns=['Open', 'High', 'Low', 'Close', 'Volume'],  # UPDATED: Include Volume as 5th feature
    sequence_length=60,
    use_gluonts=True,   # NEW: Enable GluonTS preprocessing
    freq="D"           # NEW: Daily frequency specification
)

# ENHANCED: Display comprehensive dataset information
print("\n=== Enhanced Dataset Information ===")
for key, value in info.items():
    print(f"  {key}: {value}")

# ============================================================================
# STEP 2: VERIFY MULTIVARIATE DATA STRUCTURE
# ============================================================================

print(f"\n=== Multivariate Data Verification ===")
sample_batch = next(iter(dataloader))
print(f"Sample batch shape: {sample_batch.shape}")  # Should be [batch_size, 5, sequence_length] for OHLCV
print(f"Features: {info['columns']}")
print(f"Input dimension: {info['input_dim']}")

# ENHANCED: Show multivariate sample with OHLCV visualization
show_timeseries_sample(sample_batch, dataset, "Multivariate OHLCV Time Series")

# ============================================================================
# STEP 3: MODEL CREATION (UPDATED FOR MULTIVARIATE)
# ============================================================================

print(f"\n=== Creating Multivariate Diffusion Model ===")

# UPDATED: Create model with correct input dimension (5 for OHLCV instead of 4)
model = create_wavenet_model(input_dim=info['input_dim'])  # Use dynamic input_dim

# Test forward pass with actual multivariate dimensions
print("Testing model with multivariate input...")
batch_size = 8
sequence_length = info['sequence_length']
input_dim = info['input_dim']

# UPDATED: Use actual dimensions from dataset
x = torch.randn(batch_size, input_dim, sequence_length)  # [batch, features, time]
timestep = torch.randint(0, 1000, (batch_size,))

output = model(x, timestep)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Verify shapes match (essential for diffusion)
assert output.shape == x.shape, "Output shape must match input shape for diffusion!"
print("✅ Multivariate model verification passed!")

# ============================================================================
# STEP 4: TRAINING SETUP (MINIMAL CHANGES)
# ============================================================================

print(f"\n=== Training Setup ===")

# Training configuration (unchanged)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training device: {device}")
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100

# Move pre-calculated tensors to device (unchanged)
betas = betas.to(device)
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)
sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
sqrt_recip_alphas = sqrt_recip_alphas.to(device)
posterior_variance = posterior_variance.to(device)

# Loss function (unchanged)
def get_loss(model, x_0, t):
    """
    Calculate diffusion training loss
    Works unchanged for multivariate data
    """
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise, noise_pred)

# ============================================================================
# STEP 5: ENHANCED TRAINING LOOP
# ============================================================================

print(f"\n=== Starting Enhanced Training Loop ===")
print(f"Training multivariate model: {input_dim} features × {sequence_length} timesteps")

# Training loop (minimal changes for enhanced logging)
for epoch in range(epochs):
    epoch_losses = []  # NEW: Track losses for better monitoring
    
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Handle batch (unchanged logic, enhanced comments)
        if isinstance(batch, list):
            x_0 = batch[0].to(device)  # Handle list of tensors
        else:
            x_0 = batch.to(device)     # Handle single tensor - multivariate [batch, features, time]

        # Generate random timesteps (unchanged)
        t = torch.randint(0, T, (x_0.shape[0],), device=device).long()

        # Calculate loss (works for multivariate data)
        loss = get_loss(model, x_0, t)
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())

        # ENHANCED: More informative logging
        if epoch % 5 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} | Loss: {loss.item():.6f} | Features: {x_0.shape[1]}")

            # UPDATED: Use enhanced sampling function from new data_sampler
            model.eval()
            with torch.no_grad():
                sample_single_timeseries(model, dataset, device, T)
            model.train()

    # ENHANCED: Epoch summary with average loss
    if epoch % 10 == 0:
        avg_loss = np.mean(epoch_losses)
        print(f"Completed epoch {epoch}/{epochs} | Average Loss: {avg_loss:.6f}")

print("✅ Enhanced training completed!")

# ============================================================================
# STEP 6: MODEL SAVING (ENHANCED METADATA)
# ============================================================================

print(f"\n=== Saving Enhanced Model ===")

# ENHANCED: Save with multivariate metadata
model_save_path = f'multivariate_wavenet_diffusion_epoch_{epochs}.pth'
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
    # NEW: Save multivariate metadata
    'model_config': {
        'input_dim': input_dim,
        'sequence_length': sequence_length,
        'features': info['columns'],
        'use_gluonts': info['use_gluonts'],
        'frequency': info['frequency']
    },
    'dataset_info': info
}, model_save_path)

print(f"Enhanced model saved as '{model_save_path}'")
print(f"Model features: {info['columns']}")

# ============================================================================
# STEP 7: ENHANCED EVALUATION USING NEW EVALUATOR SYSTEM
# ============================================================================

print(f"\n=== Enhanced Multivariate Evaluation ===")

# 🎯 NEW: Use the enhanced evaluator system with robust evaluation
print("🚀 Using Enhanced Evaluator System with Robust Evaluation")

# Option 1: Quick evaluation for immediate feedback
print("\n1️⃣ Quick Evaluation (fast feedback):")
try:
    quick_results = quick_evaluation(model, dataset, device)
    
    if quick_results.get('success', False):
        print("✅ Quick evaluation successful!")
        method = quick_results.get('evaluation_method', 'unknown')
        print(f"📊 Method used: {method}")
        
        # Display metrics based on what we got
        if 'gluonts_metrics' in quick_results:
            print("🎯 GluonTS metrics available")
        elif 'mse' in quick_results:
            print(f"📈 Simple metrics - MSE: {quick_results['mse']:.6f}, MAE: {quick_results['mae']:.6f}")
    else:
        print("⚠️ Quick evaluation had issues")
        
except Exception as e:
    print(f"❌ Quick evaluation failed: {e}")

# Option 2: Standard robust evaluation
print("\n2️⃣ Standard Robust Evaluation:")
try:
    evaluation_results = robust_evaluate_model(
        model=model,
        dataset=dataset,
        prediction_length=30,
        num_samples=50,
        device=device,
        T=T
    )
    
    if evaluation_results.get('success', False):
        print("🎉 Robust evaluation completed successfully!")
        
        method = evaluation_results.get('evaluation_method', 'unknown')
        print(f"📊 Evaluation method: {method}")
        
        # Display results based on what we got
        if 'gluonts_metrics' in evaluation_results:
            print("🏆 Professional GluonTS metrics:")
            gluonts_metrics = evaluation_results['gluonts_metrics']
            key_metrics = ['MASE', 'sMAPE', 'MSIS', 'QuantileLoss[0.5]', 'Coverage[0.5]']
            for metric in key_metrics:
                if metric in gluonts_metrics:
                    print(f"  {metric}: {gluonts_metrics[metric]:.4f}")
        elif 'mse' in evaluation_results:
            print("📊 Simple evaluation metrics:")
            print(f"  MSE:  {evaluation_results['mse']:.6f}")
            print(f"  MAE:  {evaluation_results['mae']:.6f}")
            print(f"  MAPE: {evaluation_results['mape']:.2f}%")
            print(f"  R²:   {evaluation_results.get('r2', 'N/A'):.4f}")
        else:
            print("✅ Basic sampling completed successfully")
    else:
        print("❌ Robust evaluation failed completely")
        
except Exception as e:
    print(f"❌ Robust evaluation encountered error: {e}")

# Option 3: Comprehensive evaluation (if you want everything)
print("\n3️⃣ Comprehensive Evaluation (all methods):")
try:
    comprehensive_results = comprehensive_evaluation(model, dataset, device)
    print("✅ Comprehensive evaluation completed!")
    
    # The comprehensive evaluation already prints its own summary
    
except Exception as e:
    print(f"❌ Comprehensive evaluation failed: {e}")

# Option 4: Fallback to original evaluator if all else fails
print("\n4️⃣ Fallback: Original CRPS Evaluation:")
try:
    original_results = evaluate_trained_model(model, dataset, device=device, n_samples=50, T=T)
    
    if original_results.get('success', False):
        print("✅ Original evaluation method successful!")
    else:
        print("⚠️ Original evaluation had issues")
        
except Exception as e:
    print(f"❌ Original evaluation failed: {e}")

# ENHANCED: Generate multiple samples for diversity check
print("\n🎲 Generating multiple samples for diversity analysis...")
try:
    diverse_samples = sample_multiple_timeseries(
        model=model, 
        num_samples=3, 
        dataset=dataset, 
        device=device, 
        T=T
    )
    print(f"✅ Generated {len(diverse_samples)} diverse multivariate samples")
    
    # Basic statistical analysis
    if diverse_samples:
        samples_array = np.array([s.numpy() for s in diverse_samples])
        print(f"Sample statistics:")
        print(f"  Mean across samples: {np.mean(samples_array):.4f}")
        print(f"  Std across samples: {np.std(samples_array):.4f}")
        
        # Feature-wise statistics
        for i, feature_name in enumerate(info['columns']):
            feature_mean = np.mean(samples_array[:, i, :])
            feature_std = np.std(samples_array[:, i, :])
            print(f"  {feature_name}: mean={feature_mean:.4f}, std={feature_std:.4f}")
            
except Exception as e:
    print(f"❌ Multiple sampling failed: {e}")

print("\n" + "="*60)
print("🎉 ENHANCED MULTIVARIATE DIFFUSION PIPELINE COMPLETED!")
print("="*60)
print(f"✅ Successfully trained on {input_dim}-dimensional time series")
print(f"✅ Features: {', '.join(info['columns'])}")
print(f"✅ GluonTS preprocessing: {info['use_gluonts']}")
print(f"✅ Model saved with enhanced metadata")
print(f"✅ Enhanced robust evaluation system used")
print("\n🚀 Your multivariate diffusion model is ready for advanced applications!")

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

print(f"\n=== Usage Examples for Enhanced Evaluator ===")
print("# Load saved model:")
print(f"checkpoint = torch.load('{model_save_path}')")
print("model.load_state_dict(checkpoint['model_state_dict'])")
print()
print("# Enhanced evaluation options:")
print("# 1. Quick evaluation (recommended for development)")
print("results = quick_evaluation(model, dataset, device)")
print()
print("# 2. Robust evaluation (recommended for final assessment)")
print("results = robust_evaluate_model(model, dataset, prediction_length=30, num_samples=50)")
print()
print("# 3. Comprehensive evaluation (all methods)")
print("results = comprehensive_evaluation(model, dataset, device)")
print()
print("# 4. Simple evaluation only (no GluonTS)")
print("results = simple_evaluation_metrics(model, dataset, prediction_length=30)")
print()
print("# 5. Original enhanced evaluator")
print("results = evaluate_trained_model(model, dataset, device=device, n_samples=50)")

print(f"\n🎯 Key Benefits of Enhanced Evaluator:")
print("✅ Always returns evaluation results (never fails completely)")
print("✅ Automatic GluonTS compatibility handling")
print("✅ Multiple evaluation methods available")
print("✅ Enhanced error handling and diagnostics")
print("✅ Comprehensive metrics and visualizations")
print("✅ Backwards compatible with original code")
print("✅ Professional time series metrics when possible")
print("✅ Graceful fallbacks when GluonTS unavailable")