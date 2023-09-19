from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass
from collections import heapify, heappush, heappop

import numpy as np


@dataclass
class HeapItem:
    sim: float
    node: Node

    @staticmethod
    def from_node_query(node: Node, query: np.ndarray) -> HeapItem:
        return HeapItem(np.linalg.norm(node.vec.dot(query)), node)


class MinHeapItem(HeapItem):
    def __lt__(self, other: MinHeapItem) -> bool:
        return self.sim < other.sim


class MaxHeapItem(HeapItem):
    def __lt__(self, other: MaxHeapItem) -> bool:
        return self.sim > other.sim


@dataclass
class Node:
    id: int
    vec: np.ndarray
    neighbors: Dict[int, List[Node]]

    @staticmethod
    def from_vec(id: int, vec: np.ndarray, nlayers: int) -> Node:
        return Node(id, vec, [[]] * nlayers)  # TODO fix the dict construction

    def shrink_connections(self, layer: int, max_connections: int) -> None:
        if len(self.neighbors[layer]) <= max_connections:
            return
        else:
            raise NotImplementedError("Implement shrinks!")

    def get_top_layer(self) -> int:
        return max(self.neighbors.keys())

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class HNSWGraphConfig:
    nlayers: int
    k_construction: int
    neighbor_max_degree: int


class HNSWGraph:
    def __init__(self, config: HNSWGraphConfig) -> None:
        self.config = config
        self.entrypoint: Node

    def insert(self, node: Node) -> None:
        insert_layer = self._sample_insert_layer()
        ep = [self.entrypoint]
        max_layer = self.entrypoint.get_top_layer()
        # first we search down to the first layer where we'll insert the noe
        for layer_idx in range(max_layer, insert_layer + 1, -1):
            ep = self._search_layer(node.vec, ep, layer_idx, 1)[
                0
            ]  # get the closest element from the greedy search
        remaining = min(max_layer, insert_layer)
        for layer_idx in range(remaining, -1, -1):
            candidates = self._search_layer(
                node.vec, ep, layer_idx, self.config.k_construction
            )
            neighbors = self._select_neighbors(
                node.vec, candidates, self.config.k_construction
            )
            for neighbor in neighbors:
                neighbor.neighbors[layer_idx].append(node)
                node.neighbors[layer_idx].append(neighbor)
                node.shrink_connections(self.config.neighbor_max_degree)
            ep = candidates
        if max_layer < insert_layer:
            self.entrypoint = node

    def search(self, query: np.ndarray, k: int) -> List[Node]:
        ep = [self.entrypoint]
        max_layer = self.entrypoint.get_top_layer()
        for layer_idx in range(max_layer, 0, -1):
            ep = self._search_layer(query, ep, layer_idx, 1)[0]
        candidates = self._search_layer(query, ep, 0, k)
        return candidates

    def _sample_insert_layer(self) -> int:
        return np.floor(-self.config.nlayers * np.log(np.random.uniform(0, 1)))

    def _search_layer(
        self, query: np.ndarray, ep: List[Node], layer: int, k: int
    ) -> List[Node]:
        visited = set([ep])
        candidates = heapify([MinHeapItem.from_node_query(node, query) for node in ep])
        best_k = heapify([MaxHeapItem.from_node_query(node, query) for node in ep])
        while len(candidates) > 0:
            cand = heappop(candidates)
            current_worst = heappop(best_k)  # we just need to peek
            heappush(best_k, current_worst)
            if cand.sim < current_worst.sim:
                # this is a greedy search, we're done
                break
            for neighbor in cand.node.neighbors[layer]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if (
                        current_worst.sim < neighbor.vec.dot(query)
                        or len(best_k) < k
                    ):
                        heappush(
                            candidates, MinHeapItem.from_node_query(neighbor, query)
                        )
                        heappush(best_k, MaxHeapItem.from_node_query(neighbor, query))
                        if len(best_k) > k:
                            heappop(best_k)
        return [item.node for item in best_k]

    def _select_neighbors(
        self, query: np.ndarray, candidates: List[Node], k: int
    ) -> List[Node]:
        # sort the candidates by dot product similarity
        # return the top k
        return sorted(candidates, key=lambda node: node.vec.dot(query))[:k]
