from dataclasses import dataclass, asdict

@dataclass
class OperationStats:
    size: int = 0
    visited: int = 0
    search_distance_computations: int = 0
    links_added: int = 0
    shrinks_distance_computations: int = 0

    def to_dict(self):
        return asdict(self)
