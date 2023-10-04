from __future__ import annotations
from enum import Enum
from typing import Callable, Dict, List, Tuple
from dataclasses import dataclass, field
from heapq import heappush, heappop

import numpy as np


def batch_apply_distance(
    distance_func: Callable, query: np.ndarray, candidates: List[Node]
) -> List[Tuple[Node, float]]:
    if len(candidates) == 0:
        return []
    matrix = np.stack([node.vec for node in candidates])
    distances = distance_func(query, matrix).tolist()
    return [(node, distance) for node, distance in zip(candidates, distances)]


@dataclass
class HeapItem:
    distance: float = field(compare=False)
    node: Node = field(compare=False)

    @classmethod
    def from_nodes_query(
        cls,
        nodes: List[Node],
        query: np.ndarray,
        distance_func: Callable,
    ) -> List[HeapItem]:
        nodes_distance_tuples = batch_apply_distance(distance_func, query, nodes)
        return [cls(distance, node) for node, distance in nodes_distance_tuples]

    def __repr__(self) -> str:
        return f"({self.distance}, {self.node.id})"


class HeapType(Enum):
    MIN = 1
    MAX = 2


class DistanceHeap:
    def __init__(self, htype: HeapType) -> None:
        self.htype = htype
        self.data = []

    def insert(self, item: HeapItem) -> None:
        if self.htype == HeapType.MIN:
            heappush(self.data, (item.distance, item))
        else:
            heappush(self.data, (-item.distance, item))

    def peek(self) -> HeapItem:
        _, node = self.data[0]
        return node

    def pop(self) -> HeapItem:
        _, node = heappop(self.data)
        return node

    def __len__(self) -> int:
        return len(self.data)

    def get_data(self) -> List[Node]:
        items = [item for _, item in self.data]
        sorted_items = sorted(items, key=lambda item: item.distance)
        return list(map(lambda item: item.node, sorted_items))


@dataclass
class Node:
    id: int
    vec: np.ndarray
    neighbors: Dict[int, List[Node]]

    @staticmethod
    def from_vec(id: int, vec: np.ndarray, nlayers: int) -> Node:
        return Node(id, vec, {layer: [] for layer in range(nlayers, -1, -1)})

    def shrink_connections(
        self, layer: int, max_connections: int, distance_func: Callable
    ) -> None:
        if len(self.neighbors[layer]) <= max_connections:
            return
        else:
            candidates = self.neighbors[layer]
            distances = batch_apply_distance(distance_func, self.vec, candidates)
            best = sorted(
                distances,
                key=lambda node: node[1],
            )[:max_connections]
            self.neighbors[layer] = [node for node, _ in best]

    def get_top_layer(self) -> int:
        return max(self.neighbors.keys())

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class HNSWGraphConfig:
    M: int
    k_construction: int  # efConstruction
    k_search: int  # efSearch

    @property
    def layer_multiplier(self) -> int:
        val = 1 / np.log(self.M)
        return val

    @property
    def max_layer0_neighbors(self) -> int:
        return self.M * 2

    @property
    def neighbor_max_degree(self) -> int:
        return self.M


class HNSWGraph:
    def __init__(self, config: HNSWGraphConfig, distance_func: Callable) -> None:
        self.config = config
        self.entrypoint: Node = None
        self.distance_func = distance_func
        self.nodes: Dict[int, Node] = dict()

    def get(self, id: int) -> Node:
        return self.nodes[id]

    def insert(self, id: int, vec: np.ndarray) -> None:
        links_added = 0
        insert_layer = self._sample_insert_layer()
        if self.entrypoint is None:
            node = Node.from_vec(id, vec, insert_layer)
            self.entrypoint = node
            return

        node = Node.from_vec(id, vec, insert_layer)
        ep = [self.entrypoint]
        max_layer = self.entrypoint.get_top_layer()
        # first we search down to the first layer where we'll insert the noe
        for layer_idx in range(max_layer, insert_layer + 1, -1):
            ep = self._search_layer(node.vec, ep, layer_idx, 1)[
                :1
            ]  # get the closest element from the greedy search

        remaining = min(max_layer, insert_layer)
        for layer_idx in range(remaining, -1, -1):
            candidates = self._search_layer(
                node.vec, ep, layer_idx, self.config.k_construction
            )
            neighbors = self._select_neighbors(node.vec, candidates, self.config.M)
            for neighbor in neighbors:
                links_added += 2
                neighbor.neighbors[layer_idx].append(node)
                node.neighbors[layer_idx].append(neighbor)
                neighbor.shrink_connections(
                    layer_idx, self.config.neighbor_max_degree, self.distance_func
                )
            ep = candidates
        if max_layer < insert_layer:
            self.entrypoint = node
        self.nodes[id] = node

    def search(self, query: np.ndarray, k: int) -> List[Node]:
        ep = [self.entrypoint]
        max_layer = self.entrypoint.get_top_layer()
        for layer_idx in range(max_layer, 0, -1):
            ep = self._search_layer(query, ep, layer_idx, self.config.k_search)[:1]
        candidates = self._search_layer(query, ep, 0, self.config.k_search)
        return candidates[:k]

    def _sample_insert_layer(self) -> int:
        return np.floor(
            -self.config.layer_multiplier * np.log(np.random.uniform(0, 1))
        ).astype(int)

    def _search_layer(
        self, query: np.ndarray, ep: List[Node], layer: int, k: int
    ) -> List[Node]:
        visited = set(ep)
        candidates = DistanceHeap(HeapType.MIN)
        best_k = DistanceHeap(HeapType.MAX)
        heap_items = HeapItem.from_nodes_query(ep, query, self.distance_func)
        for item in heap_items:
            candidates.insert(item)
            best_k.insert(item)
        while len(candidates) > 0:
            cand = candidates.pop()
            current_worst = best_k.peek()
            if cand.distance > current_worst.distance:
                # this is a greedy search, we're done
                break
            neighbors = set(cand.node.neighbors[layer])
            to_visit = neighbors.difference(visited)
            to_visit_heap_items = HeapItem.from_nodes_query(
                list(to_visit), query, self.distance_func
            )
            for provisional in to_visit_heap_items:
                visited.add(provisional.node)
                if (
                    current_worst.distance >= provisional.distance
                    or len(best_k) < k
                ):
                    candidates.insert(provisional)
                    best_k.insert(provisional)
                    if len(best_k) > k:
                        best_k.pop()
        return best_k.get_data()

    def _select_neighbors(
        self, query: np.ndarray, candidates: List[Node], k: int
    ) -> List[Node]:
        # sort the candidates by dot product similarity
        # return the top k
        distances = batch_apply_distance(self.distance_func, query, candidates)
        best = sorted(
            distances,
            key=lambda node: node[1],
        )[:k]
        return [node for node, _ in best]
