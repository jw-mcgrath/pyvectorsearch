from impls.optimized.hnsw_base import HNSWGraphConfig, HNSWGraph
import numpy as np
import torch
import pandas as pd
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

def profile_custom_hnsw(data, M=30, efConstruction=100, efSearch=10):
    def distance_func(query, candidates):
        return torch.linalg.norm(query - candidates, dim=1)

    config = HNSWGraphConfig(k_construction=efConstruction, M=M, k_search=efSearch)
    graph = HNSWGraph(config, distance_func=distance_func)
    stats = []
    for i, vec in enumerate(data):
        stat = graph.insert(i, vec)
        stats.append(stat)
    return list(map(lambda x: x.to_dict(), stats))
    


def profile():
    data = generate_unit_sphere_vectors(1000, 128)
    print("Creating a line profile of the insert...")
    with cProfile.Profile() as pr:
        setup_custom_hnsw(data)
    pr.dump_stats("exps/output_data/insert_profiler.prof")
    print("Generating a dump of insert stats...")
    stats = profile_custom_hnsw(data)
    df = pd.DataFrame(stats)
    df.to_csv("exps/output_data/insert_stats.csv")


if __name__ == "__main__":
    profile()
