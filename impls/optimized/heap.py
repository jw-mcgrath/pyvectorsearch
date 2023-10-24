from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from typing import List, Callable, Set, Dict, Tuple

from impls.optimized.node import Node


class HeapType(Enum):
    MIN = 1
    MAX = 2


@dataclass
class HeapItem:
    distance: float = field(compare=False)
    node: Node = field(compare=False)

    def __repr__(self) -> str:
        return f"({self.distance}, {self.node.id})"


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
