from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List

import torch

from impls.optimized.utils import batch_apply_distance


@dataclass
class Node:
    id: int
    vec: torch.Tensor
    neighbors: Dict[int, List[Node]]
    
    @staticmethod
    def from_vec(id: int, vec: torch.Tensor, nlayers: int) -> Node:
        return Node(id, vec, {layer: [] for layer in range(nlayers, -1, -1)})

    def shrink_connections(
        self, layer: int, max_connections: int, distance_func: Callable
    ) -> bool:
        if len(self.neighbors[layer]) <= max_connections:
            return False
        else:
            candidates = self.neighbors[layer]
            distances = batch_apply_distance(distance_func, self.vec, candidates)
            best = sorted(
                distances,
                key=lambda node: node[1],
            )[:max_connections]
            self.neighbors[layer] = [node for node, _ in best]
            return True

    def get_top_layer(self) -> int:
        return max(self.neighbors.keys())

    def __hash__(self) -> int:
        return hash(self.id)
