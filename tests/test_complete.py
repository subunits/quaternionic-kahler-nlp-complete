"""Complete test suite for Quaternionic-Kähler Natural Language Processing."""

import pytest
import torch
import math
import sys
sys.path.insert(0, '/home/claude/quaternionic-kahler-nlp-complete')

from src.quaternion_nlp import (
    Quaternion,
    QuaternionEmbedding,
    QuaternionLinear,
    quaternion_normalize,
    quaternion_batch_norm,
    KahlerMetric,
    QuaternionicAttention,
    QuaternionicTransformer,
)


class TestQuaternion:
    """Tests for quaternion operations."""
    
    def test_quaternion_creation(self):
        """Test quaternion initialization."""
        q = Quaternion(1, 2, 3, 4)
        assert q.a == 1 and q.b == 2 and q.c == 3 and q.d == 4
    
    def test_quaternion_conjugate(self):
        """Test quaternion conjugation."""
        q = Quaternion(1, 2, 3, 4)
        q_conj = q.conjugate()
        assert q_conj.a == 1 and q_conj.b == -2
    
    def test_quaternion_norm(self):
        """Test quaternion norm."""
        q = Quaternion(3, 4, 0, 0)
        assert abs(q.norm() - 5.0) < 1e-6
    
    def test_quaternion_normalize(self):
        """Test quaternion normalization."""
        q = Quaternion(3, 4, 0, 0)
        q_norm = q.normalize()
        assert abs(q_norm.norm() - 1.0) < 1e-6
    
    def test_hamilton_product_norm(self):
        """Test norm multiplicative property."""
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(5, 6, 7, 8)
        product = q1.hamilton_product(q2)
        expected = q1.norm() * q2.norm()
        assert abs(product.norm() - expected) < 1e-6


class TestQuaternionEmbedding:
    """Tests for quaternionic embedding layer."""
    
    def test_embedding_shape(self):
        """Test embedding output shape."""
        embedding = QuaternionEmbedding(vocab_size=100, embedding_dim=32)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        output = embedding(input_ids)
        assert output.shape == (2, 3, 32)
    
    def test_embedding_normalize(self):
        """Test embedding normalization."""
        embedding = QuaternionEmbedding(vocab_size=100, embedding_dim=32)
        embedding.normalize_embeddings()
        norms = torch.norm(embedding.embedding.weight, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


class TestQuaternionicTransformer:
    """Tests for quaternionic transformer."""
    
    def test_transformer_forward(self):
        """Test transformer forward pass."""
        model = QuaternionicTransformer(
            vocab_size=1000,
            embedding_dim=64,
            num_layers=2,
            num_heads=4,
            hidden_dim=128,
        )
        
        input_ids = torch.randint(0, 1000, (2, 20))
        output, attention_weights = model(input_ids)
        
        assert output.shape == (2, 20, 64)
        assert len(attention_weights) == 2
    
    def test_transformer_sentence_embedding(self):
        """Test transformer sentence embedding."""
        model = QuaternionicTransformer(
            vocab_size=1000,
            embedding_dim=64,
            num_layers=2,
        )
        
        input_ids = torch.randint(0, 1000, (2, 20))
        embeddings = model.get_sentence_embedding(input_ids)
        
        assert embeddings.shape == (2, 64)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline."""
        model = QuaternionicTransformer(
            vocab_size=1000,
            embedding_dim=64,
            num_layers=2,
            num_heads=4,
        )
        
        input_ids = torch.randint(0, 1000, (4, 15))
        output, _ = model(input_ids)
        sent_emb = model.get_sentence_embedding(input_ids)
        
        assert output.shape == (4, 15, 64)
        assert sent_emb.shape == (4, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
