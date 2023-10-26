from impls.optimized.hnsw_base import HNSWGraph
from impls.optimized.op_stats import OperationStats
from impls.optimized.utils import batch_apply_distance


class LocalReconnectHNSWGraph(HNSWGraph):
    def delete(self, id: int) -> OperationStats:
        self._reset_op_stats()
        node_to_delete = self.nodes[id]
        for node in self.nodes.values():
            for layer in node.neighbors.keys():
                if node_to_delete in node.neighbors[layer]:
                    local_neighbors = set(node_to_delete.neighbors[layer])
                    to_remove_candidates = set([node] + node.neighbors[layer])
                    local_candidates = list(local_neighbors.difference(to_remove_candidates))
                    if len(local_candidates) > 0:
                        distances = batch_apply_distance(
                            self.distance_func, node.vec, local_candidates
                        )
                        best = sorted(
                            distances,
                            key=lambda node: node[1],
                        )
                        node.neighbors[layer].append(best[0][0])
                    node.neighbors[layer].remove(node_to_delete)