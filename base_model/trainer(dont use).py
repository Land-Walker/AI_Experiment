from torch.optim import Adam
import torch

# Training setup - same as your original code
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100  # Try more!

# Move pre-calculated tensors to the correct device
betas = betas.to(device)
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)
sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
sqrt_recip_alphas = sqrt_recip_alphas.to(device)
posterior_variance = posterior_variance.to(device)

# Make sure you have the loss function (should work unchanged)
def get_loss(model, x_0, t):
    """
    Calculate the loss for diffusion training.
    This function should work unchanged for time series.
    """
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise, noise_pred)

# Training loop - almost identical to your original
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Move batch to device and get correct shape
        if isinstance(batch, list):
            x_0 = batch[0].to(device)  # Handle list of tensors
        else:
            x_0 = batch.to(device)     # Handle single tensor

        # x_0 should now be [batch, features, time] from the updated dataset
        # No need for reshaping since dataset now returns correct format

        # Generate random timesteps
        t = torch.randint(0, T, (x_0.shape[0],), device=device).long()  # Use actual batch size

        # Calculate loss (same as original)
        loss = get_loss(model, x_0, t)
        loss.backward()
        optimizer.step()

        # Logging and sampling (modified for time series)
        if epoch % 5 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item():.6f}")

            # Sample and show time series instead of image
            model.eval()
            with torch.no_grad():
                # Pass the model to sample_single_timeseries
                sample_single_timeseries(model, dataset, device, T)
            model.train()

    # Optional: Print epoch summary
    if epoch % 10 == 0:
        print(f"Completed epoch {epoch}/{epochs}")

print("Training completed!")

# Optional: Save the trained model
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}, f'wavenet_diffusion_epoch_{epochs}.pth')

print(f"Model saved as 'wavenet_diffusion_epoch_{epochs}.pth'")
print("Starting evaluation...")
results = evaluate_trained_model(model, dataset, device=device, n_samples=50, T=T)