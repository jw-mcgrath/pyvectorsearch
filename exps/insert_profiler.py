from impls.deletes.hnsw_base import HNSWGraphConfig, HNSWGraph
import numpy as np
import torch
import cProfile


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
    return torch.from_numpy(normalized_vectors)


def setup_custom_hnsw(data, M=30, efConstruction=100, efSearch=10):
    def distance_func(query, candidates):
        return torch.linalg.norm(query - candidates, dim=1)

    config = HNSWGraphConfig(k_construction=efConstruction, M=M, k_search=efSearch)
    graph = HNSWGraph(config, distance_func=distance_func)
    for i, vec in enumerate(data):
        graph.insert(i, vec)
    return graph


def profile():
    data = generate_unit_sphere_vectors(1000, 128)
    with cProfile.Profile() as pr:
        setup_custom_hnsw(data)
    pr.dump_stats("exps/output_data/insert_profiler.prof")


if __name__ == "__main__":
    profile()
