"""Utility functions for quaternionic Natural Language Processing."""

from typing import List, Dict, Any, Tuple
import json
import torch
import torch.nn as nn
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f)


def save_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: str,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)


def load_model(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, int]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch']


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_device(device_str: str = "auto") -> torch.device:
    """Get torch device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_str)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProgressTracker:
    """Track training progress."""
    
    def __init__(self, total_steps: int):
        """Initialize progress tracker."""
        self.total_steps = total_steps
        self.current_step = 0
        self.losses = []
        self.accuracies = []
    
    def update(self, loss: float, accuracy: float = None) -> None:
        """Update progress."""
        self.current_step += 1
        self.losses.append(loss)
        if accuracy is not None:
            self.accuracies.append(accuracy)
    
    def get_progress(self) -> float:
        """Get progress percentage."""
        return (self.current_step / self.total_steps) * 100
    
    def get_average_loss(self, window: int = 100) -> float:
        """Get average loss over recent window."""
        if not self.losses:
            return 0.0
        return sum(self.losses[-window:]) / min(len(self.losses), window)


class EarlyStopping:
    """Early stopping monitor."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        """Initialize early stopping."""
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """Check if should stop training."""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def create_directories(paths: List[str]) -> None:
    """Create directories if they don't exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(output_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(output_path, 'r') as f:
        return json.load(f)
