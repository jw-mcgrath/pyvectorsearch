import torch
import numpy as np

from impls.optimized.hnsw_base import HNSWGraph, HNSWGraphConfig


def brute_force_search(query, data, k=10):
    """
    Perform a brute force search to find the k nearest neighbors of a query point.

    Args:
    - query (np.array): The query point of shape (dimension, ).
    - data (np.array): The dataset of shape (num_points, dimension).
    - k (int): Number of neighbors to retrieve.

    Returns:
    - indices (np.array): Indices of the k nearest neighbors in the dataset.
    """
    distances = np.linalg.norm(data - query, axis=1)
    return np.argsort(distances)[:k].flatten().tolist()

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

def compute_recall(approximate_results, exact_results):
    """
    Compute the recall between approximate and exact nearest neighbor results.

    Args:
    - approximate_results (list or np.array): Results from the approximate nearest neighbor search.
    - exact_results (list or np.array): Results from the exact nearest neighbor search.

    Returns:
    - recall (float): The fraction of approximate results that match the exact results.
    """
    intersection = len(set(approximate_results) & set(exact_results))
    recall = intersection / len(exact_results)
    return recall