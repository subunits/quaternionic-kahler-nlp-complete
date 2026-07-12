"""Quaternionic-Kähler Natural Language Processing package.

A research-ready Natural Language Processing architecture combining quaternionic
embeddings and Kähler geometry-inspired attention for efficient semantic representation.
"""

__version__ = "1.0.0"
__author__ = "Quaternionic-Kähler Research Team"

from .quaternion import (
    Quaternion,
    QuaternionEmbedding,
    QuaternionLinear,
    QuaternionReLU,
    QuaternionDropout,
    quaternion_normalize,
    quaternion_batch_norm,
)

from .kahler import (
    KahlerMetric,
    QuaternionicAttention,
)

from .transformer import (
    QuaternionicFeedForward,
    QuaternionicTransformerLayer,
    QuaternionicTransformer,
)

from .utils import (
    load_config,
    save_config,
    save_model,
    load_model,
    count_parameters,
    get_device,
    set_seed,
    ProgressTracker,
    EarlyStopping,
    create_directories,
    save_results,
    load_results,
)

__all__ = [
    "Quaternion",
    "QuaternionEmbedding",
    "QuaternionLinear",
    "QuaternionReLU",
    "QuaternionDropout",
    "quaternion_normalize",
    "quaternion_batch_norm",
    "KahlerMetric",
    "QuaternionicAttention",
    "QuaternionicFeedForward",
    "QuaternionicTransformerLayer",
    "QuaternionicTransformer",
    "load_config",
    "save_config",
    "save_model",
    "load_model",
    "count_parameters",
    "get_device",
    "set_seed",
    "ProgressTracker",
    "EarlyStopping",
    "create_directories",
    "save_results",
    "load_results",
]
