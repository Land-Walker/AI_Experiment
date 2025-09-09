"""
Configuration module for time series diffusion models.
Contains all global variables and settings.
"""

# Auto-calculated based on your data (These will be set when the dataset is created)
SEQUENCE_LENGTH = None  # Will be set automatically
INPUT_DIM = None        # Will be set automatically
BATCH_SIZE = 128

# Training configuration
LEARNING_RATE = 0.001
EPOCHS = 100
T = 300  # Number of diffusion timesteps

# Model configuration
RESIDUAL_CHANNELS = 64
SKIP_CHANNELS = 64
NUM_LAYERS = 10
TIME_EMB_DIM = 32

# Data configuration
DEFAULT_TICKER = "^GSPC"
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_COLUMNS = ['Open', 'High', 'Low', 'Close']

def update_data_dimensions(sequence_length, input_dim):
    """Update global data dimensions when dataset is created"""
    global SEQUENCE_LENGTH, INPUT_DIM
    SEQUENCE_LENGTH = sequence_length
    INPUT_DIM = input_dim
    print(f"Updated dimensions: SEQUENCE_LENGTH={SEQUENCE_LENGTH}, INPUT_DIM={INPUT_DIM}")

def get_current_config():
    """Return current configuration as dictionary"""
    return {
        'SEQUENCE_LENGTH': SEQUENCE_LENGTH,
        'INPUT_DIM': INPUT_DIM,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'EPOCHS': EPOCHS,
        'T': T,
        'RESIDUAL_CHANNELS': RESIDUAL_CHANNELS,
        'SKIP_CHANNELS': SKIP_CHANNELS,
        'NUM_LAYERS': NUM_LAYERS,
        'TIME_EMB_DIM': TIME_EMB_DIM
    }