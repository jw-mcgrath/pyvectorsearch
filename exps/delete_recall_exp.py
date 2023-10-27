from typing import Dict, List

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from impls.optimized.local_reconnect import LocalReconnectHNSWGraph
from impls.optimized.naive_delete import NaiveDeleteHNSWGraph
from impls.optimized.node import Node
from tests.test_utils import (
    brute_force_search,
    compute_recall,
    generate_unit_sphere_vectors,
    setup_custom_hnsw,
)

def score(results: List[Node], expected: List[int]):
    ids = [node.id for node in results]
    return compute_recall(ids, expected)
    


def run_exp_single(graph_cls, dataset, query_set) -> Dict:
    """
    insert the dataset, get recall numbers on the query set (vs exact)
    then delete 5% of the dataset and recompute recall numbers
    """
    index = setup_custom_hnsw(graph_cls, dataset)
    initial_results = []
    for i in range(query_set.shape[0]):
        query = torch.from_numpy(query_set[i])
        results, _ = index.search(query, 10)
        expected = brute_force_search(query.numpy(), dataset, k=10)
        recall = score(results, expected)
        initial_results.append(recall)
    random_deletes = np.random.choice(
        dataset.shape[0], size=round(dataset.shape[0] * 0.01), replace=False
    )
    new_dataset = np.delete(dataset, random_deletes, axis=0)
    for i in random_deletes:
        index.delete(i)
    new_results = []
    for i in range(query_set.shape[0]):
        query = torch.from_numpy(query_set[i])
        results, _ = index.search(query, 10)
        expected = brute_force_search(query.numpy(), new_dataset, k=10)
        recall = score(results, expected)
        new_results.append(recall)
    # get the average drop in recall between queries
    diffs = np.array(initial_results) - np.array(new_results)
    return {"graph": graph_cls.__name__, "avg_recall_drop": np.mean(diffs), "size": dataset.shape[0], "variance": np.var(diffs)}


import multiprocessing

def run_exp():
    """
    run the experiment for all graph classes
    """
    graph_cls = [NaiveDeleteHNSWGraph, LocalReconnectHNSWGraph]
    results = []
    with multiprocessing.Pool() as pool:
        for size in range(10_000, 100_000, 10_000):
            for _ in range(10):
                dataset = generate_unit_sphere_vectors(size, 128)
                query_set = generate_unit_sphere_vectors(200, 128)
                for cls in graph_cls:
                    args = (cls, dataset, query_set)
                    pool.apply_async(run_exp_single, args=args, callback=results.append)
        pool.close()
        pool.join()
    df = pd.DataFrame(results)
    df.to_csv("exps/output_data/delete_recall_results.csv")


if __name__ == "__main__":
    run_exp()
