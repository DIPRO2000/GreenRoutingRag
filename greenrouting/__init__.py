"""
GreenRouting: An Energy-Aware Retrieval-Augmented Generation Framework
with Adaptive Model Routing and Quality-Aware Caching.
"""
try:
    # Pre-initialize PyTorch runtime DLLs on Windows for OpenMP / FAISS compatibility
    import torch
except ImportError:
    pass

__version__ = "1.0.0"
