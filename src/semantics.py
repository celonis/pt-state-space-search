import dataclasses
from dataclasses import dataclass
from typing import Set

from node_state import NodeState, invert_node_state
from process_tree import ProcessTree
from tree_state import ProcessTreeState
from tree_utils import is_do_child, is_parallel, is_redo_child, is_reverse_sequence, is_root, is_sequence, is_xor


@dataclass(frozen=True)
class Transition:
    node: ProcessTree
    from_state: NodeState
    to_state: NodeState

    @classmethod
    def future_to_open(cls, node: ProcessTree) -> "Transition":
        return Transition(node, NodeState.FUTURE, NodeState.OPEN)

    @classmethod
    def future_to_closed(cls, node: ProcessTree) -> "Transition":
        return Transition(node, NodeState.FUTURE, NodeState.CLOSED)

    @classmethod
    def open_to_closed(cls, node: ProcessTree) -> "Transition":
        return Transition(node, NodeState.OPEN, NodeState.CLOSED)

    @classmethod
    def closed_to_future(cls, node: ProcessTree) -> "Transition":
        return Transition(node, NodeState.CLOSED, NodeState.FUTURE)

    def is_open_to_closed(self) -> bool:
        return self.from_state == NodeState.OPEN and self.to_state == NodeState.CLOSED

    def is_future_to_open(self) -> bool:
        return self.from_state == NodeState.FUTURE and self.to_state == NodeState.OPEN

    def is_future_to_closed(self) -> bool:
        return self.from_state == NodeState.FUTURE and self.to_state == NodeState.CLOSED

    def is_closed_to_future(self) -> bool:
        return self.from_state == NodeState.CLOSED and self.to_state == NodeState.FUTURE

    def invert(self) -> "Transition":
        return dataclasses.replace(
            self,
            node=self.node,
            from_state=invert_node_state(self.to_state),
            to_state=invert_node_state(self.from_state),
        )


class ProcessTreeSemanticsInvertible:
    @classmethod
    def can_future_to_open(
        cls, tree: ProcessTree, state: ProcessTreeState[ProcessTree]
    ) -> bool:
        assert state.is_future(tree)

        if not all(state.all_descendants_future(child) for child in tree.children):
            return False

        if is_root(tree):
            return True

        if not state.is_open(tree.parent):
            return False

        return cls._future_to_open_sibling_conditions(tree, state)

    @classmethod
    def can_open_to_closed(
        cls, tree: ProcessTree, state: ProcessTreeState[ProcessTree]
    ) -> bool:
        assert state.is_open(tree)

        if not all(state.all_descendants_closed(child) for child in tree.children):
            return False

        if is_root(tree):
            return True

        if not state.is_open(tree.parent):
            return False

        return cls._open_to_closed_sibling_conditions(tree, state)

    @classmethod
    def can_future_to_closed(
        cls, tree: ProcessTree, state: ProcessTreeState[ProcessTree]
    ) -> bool:
        if not state.is_future(tree):
            return False

        if is_root(tree):
            return False

        if state.is_open(tree.parent):
            return cls._future_to_closed_sibling_conditions(tree, state)

        return cls.can_future_to_closed(tree.parent, state)

    @classmethod
    def can_closed_to_future(
        cls, tree: ProcessTree, state: ProcessTreeState[ProcessTree]
    ) -> bool:
        if not state.is_closed(tree):
            return False

        if is_root(tree):
            return False

        if state.is_open(tree.parent):
            return cls._closed_to_future_sibling_conditions(tree, state)

        return cls.can_closed_to_future(tree.parent, state)

    @classmethod
    def _closed_to_future_sibling_conditions(
        cls, tree: ProcessTree, state: ProcessTreeState[ProcessTree]
    ) -> bool:
        if is_do_child(tree):
            return state.is_open(tree.parent.children[1])

        if is_redo_child(tree):
            return not state.is_open(tree.parent.children[0])

        return False

    @classmethod
    def _future_to_closed_sibling_conditions(
        cls, tree: ProcessTree, state: ProcessTreeState[ProcessTree]
    ) -> bool:
        idx = tree.parent.children.index(tree)
        lsibs, rsibs = tree.parent.children[:idx], tree.parent.children[idx + 1 :]
        sibs = lsibs + rsibs

        if is_xor(tree.parent):
            return any(state.is_open(sib) for sib in sibs)

        if is_redo_child(tree):
            return all(state.is_open(lsib) for lsib in lsibs)

        return False

    @classmethod
    def _future_to_open_sibling_conditions(
        cls, tree: ProcessTree, state: ProcessTreeState[ProcessTree]
    ) -> bool:
        parent: ProcessTree = tree.parent

        if is_parallel(parent):
            return True

        idx = parent.children.index(tree)

        if is_sequence(parent):
            return all(
                state.all_descendants_closed(lsib) for lsib in parent.children[:idx]
            ) and all(
                state.all_descendants_future(rsib)
                for rsib in parent.children[idx + 1 :]
            )

        if is_reverse_sequence(parent):
            return all(
                state.all_descendants_future(lsib) for lsib in parent.children[:idx]
            ) and all(
                state.all_descendants_closed(rsib)
                for rsib in parent.children[idx + 1 :]
            )

        if is_xor(parent):
            return all(state.all_descendants_future(sib) for sib in parent.children)

        if is_do_child(tree):
            return state.all_descendants_future(parent.children[1])

        if is_redo_child(tree):
            return state.all_descendants_closed(parent.children[0])

        return False

    @classmethod
    def _open_to_closed_sibling_conditions(
        cls, tree: ProcessTree, state: ProcessTreeState[ProcessTree]
    ) -> bool:
        parent: ProcessTree = tree.parent

        if is_parallel(parent):
            return True

        idx = parent.children.index(tree)

        if is_sequence(parent):
            return all(
                state.all_descendants_closed(lsib) for lsib in parent.children[:idx]
            ) and all(
                state.all_descendants_future(rsib)
                for rsib in parent.children[idx + 1 :]
            )

        if is_reverse_sequence(parent):
            return all(
                state.all_descendants_future(lsib) for lsib in parent.children[:idx]
            ) and all(
                state.all_descendants_closed(rsib)
                for rsib in parent.children[idx + 1 :]
            )

        if is_xor(parent):
            return all(
                state.all_descendants_closed(sib)
                for sib in parent.children
                if sib is not tree
            )

        if is_do_child(tree):
            return state.all_descendants_closed(parent.children[1])

        if is_redo_child(tree):
            return state.all_descendants_future(parent.children[0])

        return False

    @classmethod
    def get_valid_transitions(
        cls, tree: ProcessTree, state: ProcessTreeState[ProcessTree]
    ) -> Set[Transition]:

        res = set()

        if state.is_future(tree):
            if cls.can_future_to_open(tree, state):
                res.add(Transition.future_to_open(tree))
                return res

            if cls.can_future_to_closed(tree, state):
                res.add(Transition.future_to_closed(tree))

        if state.is_closed(tree) and cls.can_closed_to_future(tree, state):
            res.add(Transition.closed_to_future(tree))

        if state.is_open(tree) and cls.can_open_to_closed(tree, state):
            res.add(Transition.open_to_closed(tree))
            return res

        for child in tree.children:
            res.update(cls.get_valid_transitions(child, state))

        return res
