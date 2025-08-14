import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from models.crnn import CRNN
from data.preprocessing import IAMDataset, collate_fn
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.visualization import plot_losses
from config import config
from utils.vocab import vocab
from torch.cuda.amp import autocast, GradScaler


def train():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CRNN for handwritten text recognition')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--resume-checkpoint', type=str, default=config.RESUME_CHECKPOINT,
                        help='Path to checkpoint to resume training')
    args = parser.parse_args()

    # Initialize device and AMP
    device = torch.device(config.DEVICE)
    scaler = GradScaler()
    print(f"Using device: {device}")

    # Create model
    model = CRNN(
        img_height=config.IMG_HEIGHT,
        num_channels=1,
        num_classes=len(config.VOCAB) + 1,
        hidden_size=config.HIDDEN_SIZE,
        num_lstm_layers=config.NUM_LSTM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)

    # Loss function with stability improvements
    criterion = nn.CTCLoss(blank=len(config.VOCAB), reduction='mean', zero_infinity=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # Load checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        model, optimizer, start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, args.resume_checkpoint, device
        )
        print(f"Resuming training from epoch {start_epoch + 1}")

    # Datasets
    train_dataset = IAMDataset(config.DATA_PATH, mode='train')
    val_dataset = IAMDataset(config.DATA_PATH, mode='val')

    # Data loaders with prefetch
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # TensorBoard writer
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Training loop
    train_losses = []
    val_losses = []
    no_improvement = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass with AMP
            with autocast():
                outputs = model(images)
                input_lengths = torch.full(
                    size=(outputs.size(1),),
                    fill_value=outputs.size(0),
                    dtype=torch.long
                ).to(device)
                target_lengths = (targets != -1).sum(dim=1).to(device)

                # NaN check
                if torch.isnan(outputs).any():
                    print("NaN in outputs! Skipping batch...")
                    optimizer.zero_grad()
                    continue

                loss = criterion(outputs, targets, input_lengths, target_lengths)

            # Skip NaN batches
            if torch.isnan(loss):
                print("NaN loss detected! Skipping batch...")
                optimizer.zero_grad()
                continue

            # Backward pass with gradient clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_train_loss += loss.item()

            # Logging
            if batch_idx % 10 == 0:
                print(
                    f"Epoch: {epoch + 1}/{args.epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                input_lengths = torch.full(
                    size=(outputs.size(1),),
                    fill_value=outputs.size(0),
                    dtype=torch.long
                ).to(device)
                target_lengths = (targets != -1).sum(dim=1).to(device)
                loss = criterion(outputs, targets, input_lengths, target_lengths)

                if torch.isnan(loss):
                    print("NaN in validation!")
                    continue

                epoch_val_loss += loss.item()

        # Save checkpoints
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        print(f"Epoch: {epoch + 1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        checkpoint_path = os.path.join(config.CHECKPOINT_PATH, f"checkpoint_epoch_{epoch + 1}.pt")
        save_checkpoint(model, optimizer, epoch, avg_val_loss, checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_PATH, "best_model.pt"))
        else:
            no_improvement += 1
            if no_improvement >= config.PATIENCE:
                print(f"No improvement for {config.PATIENCE} epochs. Early stopping...")
                break

    writer.close()
    plot_losses(train_losses, val_losses, save_path=os.path.join(log_dir, "loss_plot.png"))
    print("Training complete!")


if __name__ == "__main__":
    # Windows-safe multiprocessing
    torch.multiprocessing.freeze_support()
    train()