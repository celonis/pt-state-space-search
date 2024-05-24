from abc import ABC, abstractmethod
import dataclasses
from typing import Generic, Tuple, TypeVar

from node_state import NodeState, char_to_node_state
from process_tree import ProcessTree
from tree_utils import get_nodes_as_set


PT = TypeVar("PT", bound="ProcessTree")


class ProcessTreeState(ABC, Generic[PT]):
    @abstractmethod
    def get_state(self, node: PT) -> NodeState:
        pass

    @abstractmethod
    def update(self, node: PT, state: NodeState) -> "ProcessTreeState[PT]":
        pass

    @abstractmethod
    def invert(self) -> "ProcessTreeState[PT]":
        pass

    @classmethod
    @abstractmethod
    def get_initial_state(cls, tree: PT) -> "ProcessTreeState[PT]":
        pass

    def is_future(self, node: PT):
        return self.get_state(node) == NodeState.FUTURE

    def is_open(self, node: PT):
        return self.get_state(node) == NodeState.OPEN

    def is_closed(self, node: PT):
        return self.get_state(node) == NodeState.CLOSED

    def all_descendants_in_state(self, tree: PT, node_state: NodeState):
        return self.get_state(tree) == node_state and all(
            self.all_descendants_in_state(child, node_state) for child in tree.children
        )

    def all_descendants_future(self, tree: PT):
        return self.all_descendants_in_state(tree, NodeState.FUTURE)

    def all_descendants_closed(self, tree: PT):
        return self.all_descendants_in_state(tree, NodeState.CLOSED)



@dataclasses.dataclass(frozen=True)
class TupleTreeState(ProcessTreeState[ProcessTree]):

    state_list: Tuple[NodeState, ...]

    def get_state(self, node: ProcessTree) -> "NodeState":
        return self.state_list[node.position]

    def update(
        self, node: ProcessTree, state: NodeState
    ) -> "TupleTreeState":
        new_state_list = list(self.state_list)
        new_state_list[node.position] = state
        return dataclasses.replace(self, state_list=tuple(new_state_list))

    def invert(self) -> "TupleTreeState":
        new_state_list = [
            state if state == NodeState.OPEN else (NodeState.FUTURE if state == NodeState.CLOSED else NodeState.CLOSED)
            for state in list(self.state_list)
        ]
        return dataclasses.replace(self, state_list=tuple(new_state_list))

    @classmethod
    def get_initial_state(cls, tree: ProcessTree) -> "TupleTreeState":
        return TupleTreeState(
            state_list=tuple(NodeState.FUTURE for _ in range(len(get_nodes_as_set(tree))))
        )

    @classmethod
    def from_string(cls, state_string: str) -> "TupleTreeState":
        return TupleTreeState(state_list=tuple(char_to_node_state(char) for char in state_string))

    def __repr__(self) -> str:
        return "".join([state.value[0] for state in self.state_list])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TupleTreeState):
            return False
        return self.state_list == other.state_list

    def __hash__(self) -> int:
        return hash(self.state_list)
