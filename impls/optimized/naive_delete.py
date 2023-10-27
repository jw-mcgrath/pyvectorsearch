from impls.optimized.hnsw_base import HNSWGraph
from impls.optimized.op_stats import OperationStats


class NaiveDeleteHNSWGraph(HNSWGraph):
    def delete(self, id: int) -> OperationStats:
        self._reset_op_stats()
        node_to_delete = self.nodes[id]
        for node in self.nodes.values():
            for layer in node.neighbors.keys():
                if node_to_delete in node.neighbors[layer]:
                    node.neighbors[layer].remove(node_to_delete)
        del self.nodes[id]
        return self.op_stats
