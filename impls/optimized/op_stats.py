from dataclasses import dataclass

@dataclass
class OperationStats:
    visited: int = 0
    search_distance_computations: int = 0
    links_added: int = 0
    shrinks_distance_computations: int = 0
