import torch 
from impls.optimized.naive_delete import NaiveDeleteHNSWGraph
from tests.test_utils import generate_unit_sphere_vectors, setup_custom_hnsw


def test_naive_delete():
    insert_data = generate_unit_sphere_vectors(500, 128)
    graph = setup_custom_hnsw(NaiveDeleteHNSWGraph, insert_data)
    graph.delete(0)
    zero_vec = torch.from_numpy(insert_data[0, :])
    results, _ = graph.search(zero_vec, 10)
    ids = [node.id for node in results]
    assert 0 not in ids