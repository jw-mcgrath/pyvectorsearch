from __future__ import annotations
from typing import Callable, Dict, List, Tuple
from dataclasses import dataclass

import torch
from impls.optimized.heap import DistanceHeap, HeapItem, HeapType
from impls.optimized.node import Node

from impls.optimized.op_stats import OperationStats
from impls.optimized.utils import batch_apply_distance


@dataclass
class HNSWGraphConfig:
    M: int
    k_construction: int  # efConstruction
    k_search: int  # efSearch

    @property
    def layer_multiplier(self) -> int:
        val = 1 / torch.log(torch.Tensor([self.M]))
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
        self.op_stats: OperationStats = OperationStats()

    def get(self, id: int) -> Node:
        return self.nodes[id]

    def insert(self, id: int, vec: torch.Tensor) -> OperationStats:
        self._reset_op_stats()

        insert_layer = self._sample_insert_layer()
        node = Node.from_vec(id, vec, insert_layer)
        self.nodes[id] = node
        if self.entrypoint is None:
            self.entrypoint = node
            return self.op_stats

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
                neighbor.neighbors[layer_idx].append(node)
                node.neighbors[layer_idx].append(neighbor)
                shrinks_computed = neighbor.shrink_connections(
                    layer_idx, self.config.neighbor_max_degree, self.distance_func
                )
                if shrinks_computed:
                    self.op_stats.shrinks_distance_computations += len(neighbor.neighbors[layer_idx]) + 1
            ep = candidates
        if max_layer < insert_layer:
            self.entrypoint = node
        return self.op_stats

    def search(self, query: torch.Tensor, k: int) -> Tuple[List[Node], OperationStats]:
        self._reset_op_stats()
        ep = [self.entrypoint]
        max_layer = self.entrypoint.get_top_layer()
        for layer_idx in range(max_layer, 0, -1):
            ep = self._search_layer(query, ep, layer_idx, self.config.k_search)[:1]
        candidates = self._search_layer(query, ep, 0, self.config.k_search)
        return candidates[:k], self.op_stats

    def _sample_insert_layer(self) -> int:
        return int(torch.floor(
            -self.config.layer_multiplier * torch.log(torch.rand(1))
        ).item())

    def _search_layer(
        self, query: torch.Tensor, ep: List[Node], layer: int, k: int
    ) -> List[Node]:
        visited = set(ep)
        candidates = DistanceHeap(HeapType.MIN)
        best_k = DistanceHeap(HeapType.MAX)
        heap_items = HeapItem.from_nodes_query(ep, query, self.distance_func)
        self.op_stats.search_distance_computations += len(heap_items)
        for item in heap_items:
            candidates.insert(item)
            best_k.insert(item)
        while len(candidates) > 0:
            self.op_stats.visited += 1
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
            self.op_stats.search_distance_computations += len(to_visit_heap_items)
            for provisional in to_visit_heap_items:
                visited.add(provisional.node)
                if current_worst.distance >= provisional.distance or len(best_k) < k:
                    candidates.insert(provisional)
                    best_k.insert(provisional)
                    if len(best_k) > k:
                        best_k.pop()
        return best_k.get_data()

    def _select_neighbors(
        self, query: torch.Tensor, candidates: List[Node], k: int
    ) -> List[Node]:
        # sort the candidates by dot product similarity
        # return the top k
        distances = batch_apply_distance(self.distance_func, query, candidates)
        best = sorted(
            distances,
            key=lambda node: node[1],
        )[:k]
        return [node for node, _ in best]
    
    def _reset_op_stats(self) -> None:
        self.op_stats = OperationStats()
        self.op_stats.size = len(self.nodes)
