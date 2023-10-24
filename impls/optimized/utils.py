from typing import Callable, List, Tuple
import torch
from impls.optimized.heap import HeapItem

from impls.optimized.node import Node


def batch_apply_distance(
    distance_func: Callable, query: torch.Tensor, candidates: List[Node]
) -> List[Tuple[Node, float]]:
    if len(candidates) == 0:
        return []
    matrix = torch.stack([node.vec for node in candidates])
    distances = distance_func(query, matrix).tolist()
    return [(node, distance) for node, distance in zip(candidates, distances)]


def nodes_to_heap_items(
    nodes: List[Node],
    query: torch.Tensor,
    distance_func: Callable,
) -> List[HeapItem]:
    nodes_distance_tuples = batch_apply_distance(distance_func, query, nodes)
    return [HeapItem(distance, node) for node, distance in nodes_distance_tuples]


def distances_to_heap_items(
    tuples: Tuple[float, Node]
) -> List[HeapItem]:
    return [HeapItem(distance, node) for distance, node in tuples]
