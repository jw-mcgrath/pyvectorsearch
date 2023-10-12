from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple

import torch


@dataclass
class Node:
    id: int
    vec: torch.Tensor
    neighbors: Dict[int, List[Node]]
    neighbor_mats: Dict[int, torch.Tensor]

    @staticmethod
    def from_vec(id: int, vec: torch.Tensor, nlayers: int) -> Node:
        return Node(
            id,
            vec,
            {layer: [] for layer in range(nlayers, -1, -1)},
            {layer: None for layer in range(nlayers, -1, -1)},
        )

    def add_neighbor(self, node: Node, layer: int) -> None:
        if not self.is_empty(layer):
            self.neighbor_mats[layer] = torch.cat(
                [self.neighbor_mats[layer], node.vec.reshape(1, -1)]
            )
        else:
            self.neighbor_mats[layer] = node.vec.reshape(1, -1)
        self.neighbors[layer].append(node)

    def is_empty(self, layer: int) -> bool:
        return len(self.neighbors[layer]) == 0
    
    def remove_neighbor(self, node: Node, layer: int) -> None:
        index = self.neighbors[layer].index(node)
        if len(self.neighbors[layer]) == 1:
            self.neighbors[layer] = []
            self.neighbor_mats[layer] = None
            return
        self.neighbors[layer].pop(index)
        self.neighbor_mats[layer] = torch.cat(
            [self.neighbor_mats[layer][:index], self.neighbor_mats[layer][index + 1 :]]
        )

    def materialize_distances(
        self,
        layer: int,
        visited: Set[Node],
        query: torch.Tensor,
        distance_func: Callable,
    ) -> List[Tuple[float, Node]]:
        if self.is_empty(layer):
            return []
        candidates = self.neighbors[layer]
        unvisited_indices = [
            i for i, node in enumerate(candidates) if node not in visited
        ]
        unvisited_candidates = [candidates[i] for i in unvisited_indices]
        mat = self.neighbor_mats[layer][unvisited_indices, :]
        distances = distance_func(query, mat)
        return [
            (distance, node) for distance, node in zip(distances, unvisited_candidates)
        ]

    def shrink_connections(
        self, layer: int, max_connections: int, distance_func: Callable
    ) -> None:
        if len(self.neighbors[layer]) <= max_connections:
            return
        else:
            candidates = self.neighbors[layer]
            distances = distance_func(self.vec, self.neighbor_mats[layer])
            combined = [
                (node, distance) for node, distance in zip(candidates, distances)
            ]
            best = sorted(
                combined,
                key=lambda node: node[1],
            )[:max_connections]
            self.neighbors[layer] = [node for node, _ in best]
            self.neighbor_mats[layer] = torch.cat(
                [node.vec.reshape(1, -1) for node, _ in best]
            )

    def get_top_layer(self) -> int:
        return max(self.neighbors.keys())

    def __hash__(self) -> int:
        return hash(self.id)