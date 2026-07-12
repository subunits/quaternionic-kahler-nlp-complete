"""Complete quaternion algebra module for Natural Language Processing."""

from typing import Union, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Quaternion:
    """Complete quaternion representation and operations."""
    
    def __init__(self, a: float, b: float, c: float, d: float) -> None:
        """Initialize quaternion with four components."""
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
    
    def conjugate(self) -> "Quaternion":
        """Return conjugate: q̄ = a - bi - cj - dk."""
        return Quaternion(self.a, -self.b, -self.c, -self.d)
    
    def norm(self) -> float:
        """Return magnitude: |q| = √(a² + b² + c² + d²)."""
        return math.sqrt(self.a**2 + self.b**2 + self.c**2 + self.d**2)
    
    def normalize(self) -> "Quaternion":
        """Return unit quaternion with norm 1.0."""
        n = self.norm()
        if n == 0:
            raise ValueError("Cannot normalize zero quaternion")
        return Quaternion(self.a/n, self.b/n, self.c/n, self.d/n)
    
    def hamilton_product(self, other: "Quaternion") -> "Quaternion":
        """Compute Hamilton product (quaternion multiplication)."""
        a1, b1, c1, d1 = self.a, self.b, self.c, self.d
        a2, b2, c2, d2 = other.a, other.b, other.c, other.d
        
        a = a1*a2 - b1*b2 - c1*c2 - d1*d2
        b = a1*b2 + b1*a2 + c1*d2 - d1*c2
        c = a1*c2 - b1*d2 + c1*a2 + d1*b2
        d = a1*d2 + b1*c2 - c1*b2 + d1*a2
        
        return Quaternion(a, b, c, d)
    
    def inverse(self) -> "Quaternion":
        """Return multiplicative inverse: q⁻¹ = q̄ / |q|²."""
        n_squared = self.norm() ** 2
        if n_squared == 0:
            raise ValueError("Zero quaternion has no inverse")
        conj = self.conjugate()
        return Quaternion(conj.a/n_squared, conj.b/n_squared, 
                         conj.c/n_squared, conj.d/n_squared)
    
    def to_matrix(self) -> torch.Tensor:
        """Convert quaternion to 3x3 rotation matrix."""
        q = self.normalize()
        a, b, c, d = q.a, q.b, q.c, q.d
        
        return torch.tensor([
            [1-2*(c**2+d**2), 2*(b*c-a*d), 2*(b*d+a*c)],
            [2*(b*c+a*d), 1-2*(b**2+d**2), 2*(c*d-a*b)],
            [2*(b*d-a*c), 2*(c*d+a*b), 1-2*(b**2+c**2)]
        ], dtype=torch.float32)
    
    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """Operator overloading for multiplication."""
        return self.hamilton_product(other)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Quaternion({self.a:.4f}, {self.b:.4f}, {self.c:.4f}, {self.d:.4f})"


class QuaternionEmbedding(nn.Module):
    """Quaternionic embedding layer for PyTorch."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int = None,
        max_norm: float = None,
        norm_type: float = 2.0,
    ) -> None:
        """Initialize quaternionic embedding layer."""
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input token IDs to quaternionic space."""
        return self.embedding(input_ids)
    
    def normalize_embeddings(self) -> None:
        """Normalize all embeddings to unit quaternions."""
        with torch.no_grad():
            norms = torch.norm(self.embedding.weight, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            self.embedding.weight.div_(norms)


class QuaternionLinear(nn.Module):
    """Quaternion-aware linear transformation layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """Initialize quaternion linear layer."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_a = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_b = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_c = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_d = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features, 4))
        else:
            self.register_parameter("bias", None)
        
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """Initialize parameters using Xavier uniform initialization."""
        for weight in [self.weight_a, self.weight_b, self.weight_c, self.weight_d]:
            nn.init.xavier_uniform_(weight)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply quaternion linear transformation."""
        out_a = F.linear(input, self.weight_a)
        out_b = F.linear(input, self.weight_b)
        out_c = F.linear(input, self.weight_c)
        out_d = F.linear(input, self.weight_d)
        
        output = torch.stack([out_a, out_b, out_c, out_d], dim=-1)
        
        if self.bias is not None:
            output = output + self.bias.unsqueeze(0).unsqueeze(0)
        
        batch_size, seq_len, out_feat, _ = output.shape
        output = output.reshape(batch_size, seq_len, -1)
        
        return output


class QuaternionReLU(nn.Module):
    """Quaternion-aware activation function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply component-wise ReLU to quaternions."""
        return F.relu(x)


class QuaternionDropout(nn.Module):
    """Quaternion-aware dropout layer."""
    
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        """Initialize quaternion dropout."""
        super().__init__()
        self.p = p
        self.inplace = inplace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout to quaternions."""
        if not self.training or self.p == 0:
            return x
        
        mask = torch.bernoulli(torch.ones(x.shape[:-1]) * (1 - self.p))
        mask = mask.unsqueeze(-1).expand_as(x).to(x.device)
        
        return x * mask / (1 - self.p)


def quaternion_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize tensors to unit quaternions."""
    norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)
    return x / norms


def quaternion_batch_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Batch normalization adapted for quaternions."""
    batch_mean = x.mean(dim=(1, 2), keepdim=True)
    batch_var = ((x - batch_mean) ** 2).mean(dim=(1, 2), keepdim=True)
    
    x_norm = (x - batch_mean) / torch.sqrt(batch_var + eps)
    
    return quaternion_normalize(x_norm, eps=eps)
