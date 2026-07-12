"""Complete working examples for Quaternionic-Kähler Natural Language Processing."""

import sys
sys.path.insert(0, '/home/claude/quaternionic-kahler-nlp-complete')

import torch
from src.quaternion_nlp import QuaternionicTransformer


def example_1_basic_embedding():
    """Example 1: Basic quaternionic embeddings."""
    print("\n" + "="*70)
    print("Example 1: Basic Quaternionic Embeddings")
    print("="*70)
    
    model = QuaternionicTransformer(
        vocab_size=5000,
        embedding_dim=64,
        num_layers=3,
        num_heads=4,
        hidden_dim=128,
    )
    
    batch_size = 4
    seq_length = 20
    input_ids = torch.randint(1, 5000, (batch_size, seq_length))
    
    with torch.no_grad():
        embeddings, _ = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Mean embedding norm: {embeddings.norm(dim=-1).mean():.4f}")
    print("✓ Successfully generated quaternionic embeddings")


def example_2_sentence_embeddings():
    """Example 2: Sentence-level embeddings."""
    print("\n" + "="*70)
    print("Example 2: Sentence-Level Embeddings")
    print("="*70)
    
    model = QuaternionicTransformer(vocab_size=5000, embedding_dim=64)
    input_ids = torch.randint(1, 5000, (8, 30))
    
    with torch.no_grad():
        sent_embeddings = model.get_sentence_embedding(input_ids)
    
    print(f"Batch size: {input_ids.shape[0]}")
    print(f"Sentence embeddings shape: {sent_embeddings.shape}")
    print(f"Mean magnitude: {sent_embeddings.norm(dim=-1).mean():.4f}")
    print("✓ Generated sentence-level embeddings")


def example_3_semantic_similarity():
    """Example 3: Computing semantic similarity."""
    print("\n" + "="*70)
    print("Example 3: Semantic Similarity")
    print("="*70)
    
    model = QuaternionicTransformer(vocab_size=256, embedding_dim=32)
    model.eval()
    
    text1 = torch.tensor([[1, 2, 3, 4, 5, 6]])
    text2 = torch.tensor([[1, 2, 3, 4, 7, 8]])
    text3 = torch.tensor([[100, 101, 102, 103, 104]])
    
    with torch.no_grad():
        emb1 = model.get_sentence_embedding(text1)
        emb2 = model.get_sentence_embedding(text2)
        emb3 = model.get_sentence_embedding(text3)
    
    sim1_2 = torch.nn.functional.cosine_similarity(emb1, emb2).item()
    sim1_3 = torch.nn.functional.cosine_similarity(emb1, emb3).item()
    
    print(f"Similarity (similar texts): {sim1_2:.4f}")
    print(f"Similarity (different texts): {sim1_3:.4f}")
    print("✓ Computed semantic similarities")


def example_4_batch_processing():
    """Example 4: Efficient batch processing."""
    print("\n" + "="*70)
    print("Example 4: Batch Processing")
    print("="*70)
    
    model = QuaternionicTransformer(
        vocab_size=1000,
        embedding_dim=64,
        num_layers=2,
    )
    model.eval()
    
    batch_sizes = [1, 4, 16, 32]
    seq_length = 50
    
    print("Processing different batch sizes:")
    with torch.no_grad():
        for batch_size in batch_sizes:
            input_ids = torch.randint(1, 1000, (batch_size, seq_length))
            output, _ = model(input_ids)
            print(f"  Batch {batch_size:2d} → Output: {output.shape}")
    
    print("✓ Successfully processed multiple batch sizes")


def example_5_model_inspection():
    """Example 5: Model inspection and statistics."""
    print("\n" + "="*70)
    print("Example 5: Model Inspection")
    print("="*70)
    
    model = QuaternionicTransformer(
        vocab_size=10000,
        embedding_dim=64,
        num_layers=6,
        num_heads=8,
        hidden_dim=256,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Architecture: QuaternionicTransformer")
    print(f"Vocabulary Size: {model.vocab_size:,}")
    print(f"Embedding Dimension: {model.embedding_dim}")
    print(f"Number of Layers: {model.num_layers}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("✓ Inspected model")


def example_6_gradient_flow():
    """Example 6: Gradient flow during backprop."""
    print("\n" + "="*70)
    print("Example 6: Gradient Flow")
    print("="*70)
    
    model = QuaternionicTransformer(vocab_size=1000, embedding_dim=32)
    input_ids = torch.randint(1, 1000, (4, 20))
    
    embeddings, _ = model(input_ids)
    loss = embeddings.sum()
    loss.backward()
    
    total_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_grad += param.grad.abs().sum().item()
    
    print(f"Loss: {loss.item():.6f}")
    print(f"Total gradient magnitude: {total_grad:.6f}")
    print("✓ Verified gradient flow")


def example_7_different_metrics():
    """Example 7: Testing different Kähler metrics."""
    print("\n" + "="*70)
    print("Example 7: Different Kähler Metrics")
    print("="*70)
    
    metrics = ["euclidean", "spherical", "hyperbolic"]
    
    for metric_type in metrics:
        model = QuaternionicTransformer(
            vocab_size=500,
            embedding_dim=32,
            num_layers=1,
            metric_type=metric_type,
            curvature=0.1 if metric_type != "euclidean" else 0.0,
        )
        
        input_ids = torch.randint(1, 500, (2, 15))
        
        with torch.no_grad():
            output, _ = model(input_ids)
        
        print(f"Metric: {metric_type:12} → Output shape: {output.shape}")
    
    print("✓ Tested all metric types")


def example_8_performance_analysis():
    """Example 8: Inference performance analysis."""
    print("\n" + "="*70)
    print("Example 8: Inference Performance")
    print("="*70)
    
    import time
    
    model = QuaternionicTransformer(
        vocab_size=5000,
        embedding_dim=64,
        num_layers=4,
    )
    model.eval()
    
    with torch.no_grad():
        _ = model(torch.randint(1, 5000, (1, 10)))
    
    seq_lengths = [10, 50, 100]
    batch_size = 4
    
    print(f"Batch size: {batch_size}")
    print("Sequence Length | Time (ms)")
    print("-" * 30)
    
    with torch.no_grad():
        for seq_len in seq_lengths:
            input_ids = torch.randint(1, 5000, (batch_size, seq_len))
            
            start = time.time()
            for _ in range(5):
                _ = model(input_ids)
            elapsed = (time.time() - start) * 1000
            
            print(f"{seq_len:15} | {elapsed/5:9.3f}")
    
    print("✓ Completed performance analysis")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Quaternionic-Kähler Natural Language Processing - Complete Examples")
    print("="*70)
    
    examples = [
        example_1_basic_embedding,
        example_2_sentence_embeddings,
        example_3_semantic_similarity,
        example_4_batch_processing,
        example_5_model_inspection,
        example_6_gradient_flow,
        example_7_different_metrics,
        example_8_performance_analysis,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"✗ Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
