from typing import Callable, List, Tuple
import torch


def batch_apply_distance(
    distance_func: Callable, query: torch.Tensor, candidates: List["Node"]
) -> List[Tuple["Node", float]]:
    if len(candidates) == 0:
        return []
    matrix = torch.stack([node.vec for node in candidates])
    distances = distance_func(query, matrix).tolist()
    return [(node, distance) for node, distance in zip(candidates, distances)]
