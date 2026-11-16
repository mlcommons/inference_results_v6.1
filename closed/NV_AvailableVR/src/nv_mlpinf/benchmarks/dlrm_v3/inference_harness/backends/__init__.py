"""
Inference backends for the inference harness.
"""

from .base import DLRMBackend
from .generative_recommener_backend import GenerativeRecommenderBackend
from .hybrid_GR_backend import HybridGRBackend


__all__ = ['DLRMBackend', 'GenerativeRecommenderBackend', 'HybridGRBackend']
