import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm


from src.model import OccupancyNetwork
from src.data import get_dataloader
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train Occupancy Network')

    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Config file path')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to ShapeNet data')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max training samples (for debuging)')

    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    
    # one epoch
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch in pbar:
        images = batch['image'].to(device) # (batch_size, 3, 224, 224)
        points = batch['points'].to(device) # (batch_size, 2048, 3)
        occupancy = batch['occupancy'].to(device) # is it in the occupancy or not for each point

        optimizer.zero_grad()
        predictions = model(images, points)

        loss = criterion(predictions, occupancy)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / max(len(train_loader), 1)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            points = batch['points'].to(device)
            occupancy = batch['occupancy'].to(device)

            predictions = model(images, points)
            loss = criterion(predictions, occupancy)
            total_loss += loss.item()

    return total_loss / max(len(val_loader), 1)


def main():
    args = parse_args()

    config = load_config(args.config)

    epochs = args.epochs or config.training.num_epochs
    batch_size = args.batch_size or config.training.batch_size
    lr = args.lr or config.training.learning_rate

    print("=" * 50)
    print("Training Occupancy Network")
    print("=" * 50)
    print(f"Data root: {args.data_root}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = OccupancyNetwork(
        latent_dim=config.model.latent_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers
    ).to(device)

    print(f"Model parameters: {model.get_num_params():,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    print("Loading data...")
    train_loader = get_dataloader(
        root=args.data_root,
        split='train',
        batch_size=batch_size,
        num_points=config.training.num_points,
        max_samples=args.max_samples,
        augment=True
    )

    val_loader = get_dataloader(
        root=args.data_root,
        split='val',
        batch_size=batch_size,
        num_points=config.training.num_points,
        max_samples=args.max_samples // 10 if args.max_samples else None
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting training...")

    checkpoint = None
    try:
        for epoch in range(start_epoch, epochs):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch + 1
            )
            
            val_loss = validate(model, val_loader, criterion, device)
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {current_lr:.6f}")

            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'latent_dim': config.model.latent_dim,
                'hidden_dim': config.model.hidden_dim,
            }

            # torch.save(checkpoint, output_dir / 'last.pt')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint['best_val_loss'] = best_val_loss
                # torch.save(checkpoint, output_dir / 'best.pt')
                print(f"  New best model saved! Val Loss: {val_loss:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if checkpoint is not None:
            # torch.save(checkpoint, output_dir / 'interrupted.pt')
            print("Checkpoint saved to interrupted.pt")
        else:
            print("No checkpoint to save (interrupted before first epoch completed)")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()
