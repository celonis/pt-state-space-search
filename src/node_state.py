from enum import Enum


class NodeState(Enum):
    OPEN = "Open"
    CLOSED = "Closed"
    FUTURE = "Future"


def invert_node_state(node_state: NodeState):
    if node_state == NodeState.CLOSED:
        return NodeState.FUTURE

    if node_state == NodeState.FUTURE:
        return NodeState.CLOSED

    if node_state == NodeState.OPEN:
        return NodeState.OPEN

def char_to_node_state(node_state_char: str) -> NodeState:
    match node_state_char:
        case "c":
            return NodeState.CLOSED
        case "f":
            return NodeState.FUTURE
        case "o":
            return NodeState.OPEN
        case _:
            raise ValueError("invalid character")