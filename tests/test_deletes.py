import torch
from impls.optimized.local_reconnect import LocalReconnectHNSWGraph 
from impls.optimized.naive_delete import NaiveDeleteHNSWGraph
from tests.test_utils import generate_unit_sphere_vectors, setup_custom_hnsw


def test_delete_impls():
    insert_data = generate_unit_sphere_vectors(500, 128)
    impls = [NaiveDeleteHNSWGraph, LocalReconnectHNSWGraph]
    for impl in impls:
        graph = setup_custom_hnsw(impl, insert_data)
        graph.delete(0)
        zero_vec = torch.from_numpy(insert_data[0, :])
        results, _ = graph.search(zero_vec, 10)
        ids = [node.id for node in results]
        assert 0 not in ids, impl
    