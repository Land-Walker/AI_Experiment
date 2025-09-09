from data_loader import *
from forward_process import *
from model import *
from data_sampler import *
from evaluator import *
from trainer import *

# Data Loading
yf_df = load_yf_data()
show_yf_data(yf_df)

# Option 1: Use your yfDf with selected columns
dataset, dataloader, info = load_timeseries_dataset(
    yf_dataframe=yf_df,  # Your existing variable
    use_columns=['Open', 'High', 'Low', 'Close'],  # Exclude Volume for cleaner data
    sequence_length=60  # 60-day sequences
)

# Option 2: Use all OHLCV data from your yfDf (auto-detects numeric columns)
# dataset, dataloader, info = load_timeseries_dataset(yf_dataframe=yfDf, sequence_length=30)

# Option 3: Just closing prices from your yfDf
# dataset, dataloader, info = load_timeseries_dataset(yf_dataframe=yfDf, use_columns=['Close'], sequence_length=100)

# Print dataset information
print("Dataset Information:")
for key, value in info.items():
    print(f"  {key}: {value}")

# ---------------------------------------------------------------------------------------------------------------------------
# Sampling
# Show a sample
sample_batch = next(iter(dataloader))
print(f"\nSample batch shape: {sample_batch.shape}")
show_timeseries_sample(sample_batch, dataset, "Stock Price Time Series")

 # Option 1: Use your yfDf with selected columns
    dataset, dataloader, info = load_timeseries_dataset(
        yf_dataframe=yf_df,  # Your existing variable
        use_columns=['Open', 'High', 'Low', 'Close'],  # Exclude Volume for cleaner data
        sequence_length=60  # 60-day sequences
    )

    # Option 2: Use all OHLCV data from your yfDf (auto-detects numeric columns)
    # dataset, dataloader, info = load_timeseries_dataset(yf_dataframe=yfDf, sequence_length=30)

    # Option 3: Just closing prices from your yfDf
    # dataset, dataloader, info = load_timeseries_dataset(yf_dataframe=yfDf, use_columns=['Close'], sequence_length=100)

    # Print dataset information
    print("Dataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Show a sample
    sample_batch = next(iter(dataloader))
    print(f"\nSample batch shape: {sample_batch.shape}")
    show_timeseries_sample(sample_batch, dataset, "Stock Price Time Series")

# ---------------------------------------------------------------------------------------------------------------------------
# Training
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