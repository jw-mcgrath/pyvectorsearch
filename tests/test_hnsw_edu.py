import numpy as np
import faiss
from impls.edu.hnsw import HNSWGraphConfig, HNSWGraph
from tests.test_utils import brute_force_search, compute_recall, generate_unit_sphere_vectors





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
    def distance_func(vec1, vec2):
        return np.linalg.norm(vec1 - vec2)
    config = HNSWGraphConfig(
        k_construction=efConstruction, M=M, k_search=efSearch
    )
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



if __name__ == "__main__":
    test_hnsw()