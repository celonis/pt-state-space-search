import heapq
import sys
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar

from process_tree import ProcessTree
from semantics import ProcessTreeSemanticsInvertible, Transition
from tree_state import ProcessTreeState
from tree_utils import get_reverse_tree, is_leaf

PT = TypeVar('PT', bound='ProcessTree')

@dataclass
class SearchState:
    dist: float
    depth: int
    tree_state: ProcessTreeState[ProcessTree]
    from_start: bool
    transition: Optional[Transition] = None
    previous_valid_transitions: Set[Transition] = None
    leaf_execution: Optional[ProcessTree] = None
    parent: Optional["SearchState"] = None

    def __lt__(self, other: "SearchState") -> bool:
        return self.dist < other.dist

    def __hash__(self) -> int:
        return hash(self.tree_state)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SearchState):
            return False
        return self.tree_state == other.tree_state


class SearchStateIterator:
    def __init__(self, search_state):
        self.current_state = search_state

    def __iter__(self):
        return self

    def __next__(self) -> SearchState:
        if self.current_state is None:
            raise StopIteration
        state = self.current_state
        self.current_state = state.parent
        return state


@dataclass
class MeetingInfo:
    best_path_cost: float = sys.maxsize
    start_node: SearchState = None
    end_node: SearchState = None


@dataclass
class SearchDataStructures:
    open_set_forward: List[SearchState]
    open_set_backward: List[SearchState]
    distances_forward: Dict[ProcessTreeState[ProcessTree], Tuple[float, SearchState]]
    distances_backward: Dict[ProcessTreeState[ProcessTree], Tuple[float, SearchState]]
    meeting_info: MeetingInfo


@dataclass
class SearchStatistics:
    visited_state: int = 0


@dataclass
class SearchResult:
    cost: float
    firing_sequence: List[Transition]
    trace: Tuple[str, ...]
    leaf_sequence: List[ProcessTree]
    search_stats: SearchStatistics


@dataclass
class PtStateSpaceSearch(Generic[PT]):
    tree: PT
    state_class: Type[ProcessTreeState[PT]]

    _reverse_tree: ProcessTree = None
    _search_data_structures: SearchDataStructures = None
    _search_statistics: SearchStatistics = None

    def init_data_structures(self):
        self._reverse_tree = get_reverse_tree(self.tree)

        sds = SearchDataStructures(
            open_set_backward=[],
            open_set_forward=[],
            distances_backward={},
            distances_forward={},
            meeting_info=MeetingInfo(),
        )

        initial_start_search_state = SearchState(
            dist=0.0,
            tree_state=self.state_class.get_initial_state(self.tree),
            from_start=True,
            depth=0,
        )
        initial_end_search_state = SearchState(
            dist=0.0,
            tree_state=self.state_class.get_initial_state(self._reverse_tree),
            from_start=False,
            depth=0,
        )

        heapq.heappush(sds.open_set_forward, initial_start_search_state)
        heapq.heappush(sds.open_set_backward, initial_end_search_state)

        self._search_data_structures = sds
        self._search_statistics = SearchStatistics()

    def search(self, unidirectional=False) -> Optional[SearchResult]:
        self.init_data_structures()
        sds = self._search_data_structures
        open_set_forward: List[SearchState] = sds.open_set_forward
        open_set_backward: List[SearchState] = sds.open_set_backward
        meeting_info: MeetingInfo = sds.meeting_info

        expand_forward = True
        lock_direction = unidirectional

        # perform initial transition in both directions
        self._expand(expand_forward=True)
        self._expand(expand_forward=False)

        while len(open_set_forward) > 0 or len(open_set_backward) > 0:
            prio_queue_start = open_set_forward[0].dist if open_set_forward else 0
            prio_queue_end = open_set_backward[0].dist if open_set_backward else 0

            if meeting_info.start_node is not None:
                return self._construct_search_result(
                    meeting_info.start_node, meeting_info.end_node
                )

            if prio_queue_end + prio_queue_start >= meeting_info.best_path_cost:
                return self._construct_search_result(
                    meeting_info.start_node, meeting_info.end_node
                )

            self._expand(expand_forward=expand_forward)
            expand_forward = not expand_forward or lock_direction

        raise ValueError("Nothing found")

    def _expand(self, expand_forward: bool):
        tree = self.tree if expand_forward else self._reverse_tree

        open_set = (
            self._search_data_structures.open_set_forward
            if expand_forward
            else self._search_data_structures.open_set_backward
        )
        distance_map = (
            self._search_data_structures.distances_forward
            if expand_forward
            else self._search_data_structures.distances_backward
        )

        if not open_set:
            return

        search_state = heapq.heappop(open_set)

        cost_so_far = (
            distance_map[search_state.tree_state][0]
            if search_state.tree_state in distance_map
            else sys.maxsize
        )

        # state has already been visited
        if cost_so_far == -sys.maxsize:
            return
        self._search_statistics.visited_state += 1

        # mark as visited
        distance_map[search_state.tree_state] = (-sys.maxsize, search_state)

        enabled_transitions: Set[Transition] = ProcessTreeSemanticsInvertible.get_valid_transitions(
                tree, search_state.tree_state
            )
        
        for transition in enabled_transitions:
            cost = 1
            executed_leaf = None

            if is_leaf(transition.node):
                is_forward_exec = transition.is_future_to_open() and expand_forward
                is_backward_exec = transition.is_open_to_closed() and not expand_forward

                if is_forward_exec or is_backward_exec:
                    cost = 1
                    executed_leaf = transition.node

            new_state = SearchState(
                dist=search_state.dist + cost,
                tree_state=search_state.tree_state.update(
                    transition.node, transition.to_state
                ),
                from_start=expand_forward,
                leaf_execution=executed_leaf,
                parent=search_state,
                transition=transition,
                depth=search_state.depth + 1,
                previous_valid_transitions=enabled_transitions,
            )

            if (
                new_state.tree_state not in distance_map
                or distance_map[new_state.tree_state][0] > new_state.dist
            ):
                heapq.heappush(open_set, new_state)
                distance_map[new_state.tree_state] = (new_state.dist, new_state)
                self._check_for_match(new_state)

    def _check_for_match(self, search_state: SearchState):
        sds = self._search_data_structures
        distance_map_other_dir = (
            sds.distances_backward if search_state.from_start else sds.distances_forward
        )
        inverse_state = search_state.tree_state.invert()

        if inverse_state in distance_map_other_dir:
            match: SearchState = distance_map_other_dir[inverse_state][1]
            if match.dist + search_state.dist < sds.meeting_info.best_path_cost:
                sds.meeting_info.best_path_cost = match.dist + search_state.dist
                sds.meeting_info.start_node = (
                    search_state if search_state.from_start else match
                )
                sds.meeting_info.end_node = (
                    search_state if not search_state.from_start else match
                )

    def _construct_search_result(
        self, state_start: SearchState, state_end: SearchState
    ):
        search_state_sequence_start = list(SearchStateIterator(state_start))
        search_state_sequence_start.reverse()
        search_state_sequence_end = list(SearchStateIterator(state_end))

        firing_sequence_start = [
            state.transition
            for state in search_state_sequence_start
            if state.transition
        ]
        firing_sequence_end = [
            state.transition.invert()
            for state in search_state_sequence_end
            if state.transition
        ]

        leaf_sequence_start = [
            state.leaf_execution
            for state in search_state_sequence_start
            if state.leaf_execution
        ]
        leaf_sequence_end = [
            state.leaf_execution
            for state in search_state_sequence_end
            if state.leaf_execution
        ]

        result = SearchResult(
            cost=state_start.dist + state_end.dist,
            firing_sequence=firing_sequence_start + firing_sequence_end,
            leaf_sequence=leaf_sequence_start + leaf_sequence_end,
            trace=None,
            search_stats=self._search_statistics,
        )
        return result