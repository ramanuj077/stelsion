import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anurag_model.architecture import UpgradedExoplanetDetectorNet
from anurag_model.losses import BinaryFocalLoss
from anurag_model.dataset import ExoplanetDataset

def train_model(epochs=5, batch_size=16, lr=0.001):
    print("--- Initializing Phase 3 Training Pipeline ---")
    
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Instantiate datasets and dataloaders
    print("Generating training and validation datasets...")
    train_dataset = ExoplanetDataset(num_samples=160, inject_prob=0.5)
    val_dataset = ExoplanetDataset(num_samples=48, inject_prob=0.5)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Instantiate model, loss, and optimizer
    model = UpgradedExoplanetDetectorNet(input_len=2000, dropout=0.3, num_heads=4).to(device)
    criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Loss Function: Focal Loss (alpha=0.25, gamma=2.0)")
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        for global_x, local_x, targets in train_loader:
            global_x, local_x, targets = global_x.to(device), local_x.to(device), targets.to(device)
            
            # Forward pass
            predictions, _ = model(global_x, local_x)
            loss = criterion(predictions, targets)
            
            # Backward pass & optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * global_x.size(0)
            
            # Calculate accuracy
            preds_bin = (predictions >= 0.5).float()
            train_correct += (preds_bin == targets).sum().item()
            total_train += targets.size(0)
            
        epoch_loss = train_loss / total_train
        epoch_acc = (train_correct / total_train) * 100.0
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for global_x, local_x, targets in val_loader:
                global_x, local_x, targets = global_x.to(device), local_x.to(device), targets.to(device)
                predictions, _ = model(global_x, local_x)
                loss = criterion(predictions, targets)
                
                val_loss += loss.item() * global_x.size(0)
                preds_bin = (predictions >= 0.5).float()
                val_correct += (preds_bin == targets).sum().item()
                total_val += targets.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (val_correct / total_val) * 100.0
        
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
    # 5. Save model weights
    os.makedirs("saved_models", exist_ok=True)
    checkpoint_path = os.path.join("saved_models", "best_pytorch_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'params': {
            'input_len': 2000,
            'num_heads': 4
        }
    }, checkpoint_path)
    print(f"\n✓ Training Completed! Saved best model checkpoint to: {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AstroAI PyTorch Training Script")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
