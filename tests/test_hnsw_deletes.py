import numpy as np
import faiss
from impls.deletes.hnsw_base import (
    HNSWGraphConfig,
    HNSWGraph,
    batch_apply_distance,
    Node,
)


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


def setup_faiss_hnsw(data, M=30, efConstruction=100):
    """
    Set up FAISS HNSW index and add data to it.

    Args:
    - data (np.array): The dataset of shape (num_points, dimension).
    - M (int): Number of bi-directional links created for every new element during construction.
    - efConstruction (int): Size of the dynamic list for the nearest neighbors (used during construction).

    Returns:
    - index (faiss.Index): FAISS HNSW index with the data.
    """
    dimension = data.shape[1]
    index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efConstruction
    index.add(data)
    return index


def setup_custom_hnsw(data, M=30, efConstruction=100, efSearch=50):
    def distance_func(query, candidates):
        return np.linalg.norm(query - candidates, axis=1)

    config = HNSWGraphConfig(k_construction=efConstruction, M=M, k_search=efSearch)
    graph = HNSWGraph(config, distance_func=distance_func)
    for i, vec in enumerate(data):
        graph.insert(i, vec)
    return graph


def faiss_hnsw_search(index, query, k=10, efSearch=50):
    """
    Use the FAISS HNSW index to retrieve the k nearest neighbors of a query point.

    Args:
    - index (faiss.Index): FAISS HNSW index with the data.
    - query (np.array): The query point of shape (dimension, ).
    - k (int): Number of neighbors to retrieve.
    - efSearch (int): Size of the dynamic list for the nearest neighbors (used during search).

    Returns:
    - indices (np.array): Indices of the k nearest neighbors in the dataset.
    """
    index.hnsw.efSearch = efSearch
    vecs, neighbors = index.search(np.expand_dims(query, 0), k)
    return neighbors.flatten().tolist()


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


def test_hnsw():
    data = generate_unit_sphere_vectors(1000, 128)
    index = setup_faiss_hnsw(data)
    queries = generate_unit_sphere_vectors(100, 128)
    approximate_neighbors = faiss_hnsw_search(index, queries[0, :])
    actual_neighbors = brute_force_search(queries[0, :], data)
    recall = compute_recall(approximate_neighbors, actual_neighbors)
    assert recall >= 0.9
    graph = setup_custom_hnsw(data)
    approximate_neighbors = [node.id for node in graph.search(queries[0, :], 10)]
    recall = compute_recall(approximate_neighbors, actual_neighbors)
    assert recall >= 0.9


def test_batch_apply():
    data = generate_unit_sphere_vectors(1000, 128)
    nodes = [Node.from_vec(i, data[i, :], 0) for i in range(1000)]
    query = generate_unit_sphere_vectors(1, 128).reshape(128)

    def distance_func(query, candidates):
        return np.linalg.norm(query - candidates, axis=1)

    scores = batch_apply_distance(distance_func, query, nodes)
    assert len(scores) == 1000
    assert all([isinstance(score, tuple) for score in scores])
