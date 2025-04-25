from typing import NamedTuple


class Node(NamedTuple):
    id: str
    max_num: int
    executable: bool = True
    measureable: bool = False


class Segment(NamedTuple):
    id: str
    left: Node | Segment
    right: Node | Segment
