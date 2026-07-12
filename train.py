"""Complete training script for Quaternionic-Kähler Natural Language Processing."""

import argparse
import sys
sys.path.insert(0, '/home/claude/quaternionic-kahler-nlp-complete')

from src.quaternion_nlp import (
    QuaternionicTransformer,
    get_device,
    set_seed,
    count_parameters,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def create_dummy_dataset(num_samples=100, seq_length=50, vocab_size=1000, batch_size=32):
    """Create dummy dataset for demonstration."""
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    train_tokens = torch.randint(1, vocab_size, (train_size, seq_length))
    val_tokens = torch.randint(1, vocab_size, (val_size, seq_length))
    
    train_labels = torch.randint(0, 2, (train_size,))
    val_labels = torch.randint(0, 2, (val_size,))
    
    train_dataset = TensorDataset(train_tokens, train_labels)
    val_dataset = TensorDataset(val_tokens, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        embeddings, _ = model(input_ids)
        embeddings = embeddings.mean(dim=1)
        logits = embeddings
        
        loss = criterion(logits, labels.float().unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}")
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            embeddings, _ = model(input_ids)
            embeddings = embeddings.mean(dim=1)
            logits = embeddings
            
            loss = criterion(logits, labels.float().unsqueeze(1))
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Quaternionic-Kähler NLP model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num-samples", type=int, default=100)
    
    args = parser.parse_args()
    
    device = get_device("auto")
    set_seed(42)
    
    print("Creating model...")
    model = QuaternionicTransformer(
        vocab_size=10000,
        embedding_dim=64,
        num_layers=6,
        num_heads=8,
        hidden_dim=256,
    ).to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {device}\n")
    
    train_loader, val_loader = create_dummy_dataset(num_samples=args.num_samples)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    print("Starting training...")
    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}/5")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
