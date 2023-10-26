import torch
import numpy as np

from impls.optimized.hnsw_base import HNSWGraph, HNSWGraphConfig


def setup_custom_hnsw(cls: HNSWGraph, data, M=30, efConstruction=100, efSearch=50) -> HNSWGraph:
    def distance_func(query, candidates):
        return torch.linalg.norm(query - candidates, dim=1)

    config = HNSWGraphConfig(k_construction=efConstruction, M=M, k_search=efSearch)

    graph = cls(config, distance_func=distance_func)

    for i, vec in enumerate(torch.from_numpy(data)):
        graph.insert(i, vec)

    return graph

def generate_unit_sphere_vectors(num_points, dimension):
    """
    Generate uniformly distributed vectors on the unit sphere in M dimensions.

    Args:
    - num_points (int): Number of vectors to generate.
    - dimension (int): Dimensionality of each vector.

    Returns:
    - vectors (np.array): Generated vectors of shape (num_points, dimension), normalized on the unit sphere.
    """
    vectors = np.random.randn(num_points, dimension)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms
    return normalized_vectors