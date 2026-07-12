"""Complete quaternionic transformer model with Kähler geometry."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quaternion import (
    QuaternionEmbedding, QuaternionLinear, QuaternionReLU, 
    QuaternionDropout, quaternion_normalize
)
from .kahler import QuaternionicAttention


class QuaternionicFeedForward(nn.Module):
    """Feed-forward network for quaternionic space."""
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize feed-forward layer."""
        super().__init__()
        self.linear1 = QuaternionLinear(embedding_dim, hidden_dim)
        self.activation = QuaternionReLU()
        self.dropout1 = QuaternionDropout(dropout)
        self.linear2 = QuaternionLinear(hidden_dim, embedding_dim)
        self.dropout2 = QuaternionDropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class QuaternionicTransformerLayer(nn.Module):
    """Single quaternionic transformer layer."""
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        hidden_dim: int = 256,
        metric_type: str = "euclidean",
        curvature: float = 0.0,
        dropout: float = 0.1,
    ) -> None:
        """Initialize transformer layer."""
        super().__init__()
        
        self.attention = QuaternionicAttention(
            embedding_dim, num_heads, metric_type, curvature, dropout
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        self.feed_forward = QuaternionicFeedForward(embedding_dim, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transformer layer."""
        attn_output, attn_weights = self.attention(x, x, x, attention_mask)
        x = self.norm1(x + attn_output)
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights


class QuaternionicTransformer(nn.Module):
    """Complete quaternionic transformer model for Natural Language Processing."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        hidden_dim: int = 256,
        max_seq_length: int = 512,
        metric_type: str = "euclidean",
        curvature: float = 0.0,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        """Initialize quaternionic transformer."""
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        
        self.embeddings = QuaternionEmbedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        
        self.register_buffer(
            "positional_embeddings",
            self._get_positional_embeddings(max_seq_length, embedding_dim)
        )
        
        self.layers = nn.ModuleList([
            QuaternionicTransformerLayer(
                embedding_dim, num_heads, hidden_dim, metric_type, curvature, dropout
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
    
    def _get_positional_embeddings(
        self,
        max_seq_length: int,
        embedding_dim: int,
    ) -> torch.Tensor:
        """Generate positional embeddings using sine/cosine."""
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * 
                            -(torch.tensor(10000.0).log() / embedding_dim))
        
        pos_emb = torch.zeros(max_seq_length, embedding_dim)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        if embedding_dim % 2 == 1:
            pos_emb[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pos_emb[:, 1::2] = torch.cos(position * div_term)
        
        return pos_emb.unsqueeze(0)
    
    def _get_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input IDs."""
        batch_size, seq_length = input_ids.shape
        mask = input_ids != self.pad_token_id
        attention_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        return attention_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Forward pass through quaternionic transformer."""
        x = self.embeddings(input_ids)
        
        seq_length = input_ids.shape[1]
        x = x + self.positional_embeddings[:, :seq_length, :]
        
        x = self.embedding_dropout(x)
        
        if attention_mask is None:
            attention_mask = self._get_attention_mask(input_ids)
        
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, attention_mask)
            attention_weights.append(attn_weights)
        
        x = self.final_norm(x)
        
        return x, attention_weights
    
    def embed(self, text: str, tokenizer=None) -> torch.Tensor:
        """Embed text string to quaternionic space."""
        if tokenizer is None:
            tokens = [ord(c) % self.vocab_size for c in text]
        else:
            tokens = tokenizer(text)
        
        input_ids = torch.tensor(tokens).unsqueeze(0)
        
        with torch.no_grad():
            embeddings, _ = self.forward(input_ids)
        
        return embeddings
    
    def get_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get sentence-level embedding (mean pooling over sequence)."""
        embeddings, _ = self.forward(input_ids, attention_mask)
        
        if attention_mask is None:
            mask = input_ids != self.pad_token_id
        else:
            mask = attention_mask[:, 0, :]
        
        mask = mask.unsqueeze(-1).float()
        
        masked_embeddings = embeddings * mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        sum_mask = torch.sum(mask, dim=1)
        
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        return sum_embeddings / sum_mask
