import torch


class Config:
    # Dataset
    DATA_PATH = "./data/iam_dataset"
    IMG_HEIGHT = 32
    IMG_WIDTH = 128
    MEAN = 0.5
    STD = 0.5

    # Training
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.0008
    WEIGHT_DECAY = 1e-5
    PATIENCE = 5  # for early stopping

    # Model
    HIDDEN_SIZE = 256
    NUM_LSTM_LAYERS = 2
    DROPOUT = 0.3

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 6
    persistent_workers = True
    PIN_MEMORY = True

    # Checkpoints
    CHECKPOINT_PATH = "./checkpoints"
    RESUME_CHECKPOINT = None  # Path to .pt file to resume training

    # Vocabulary
    VOCAB = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


config = Config()