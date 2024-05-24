import re
from typing import List, Set

from process_tree import Operator, ProcessTree


def parse_tree_string(tree_string: str) -> ProcessTree:
    position_counter = [0]

    def _parse_tree_from_string(
        process_tree_string: str, parent: ProcessTree = None
    ) -> ProcessTree:
        process_tree_string = process_tree_string.strip()

        operator_regex = r"({})\((.*)\)".format(
            "|".join(re.escape(op.value) for op in list(Operator))
        )
        match = re.match(operator_regex, process_tree_string)
        if match:
            operator_value = match.group(1)
            remaining_string = match.group(2)

            operator = Operator(operator_value)

            current_position = position_counter[0]
            position_counter[0] += 1
            current_node = ProcessTree(
                position=current_position, operator=operator, parent=parent
            )

            if parent:
                parent.children.append(current_node)

            children_strings = split_arguments(remaining_string)
            for child_str in children_strings:
                _parse_tree_from_string(child_str, parent=current_node)

            return current_node
        else:
            label_match = re.match(r"'(.*?)'", process_tree_string)
            if label_match:
                label = label_match.group(1)
                current_position = position_counter[0]
                position_counter[0] += 1
                current_node = ProcessTree(
                    position=current_position, label=label, parent=parent
                )

                if parent:
                    parent.children.append(current_node)

                return current_node
            else:
                raise ValueError("Invalid tree string format")

    return _parse_tree_from_string(tree_string)


def split_arguments(arguments_string: str) -> List[str]:
    arguments = []
    nested_level = 0
    current_argument = []

    for char in arguments_string:
        if char == "(":
            nested_level += 1
        elif char == ")":
            nested_level -= 1

        if char == "," and nested_level == 0:
            arguments.append("".join(current_argument).strip())
            current_argument = []
        else:
            current_argument.append(char)

    if current_argument:
        arguments.append("".join(current_argument).strip())

    return arguments


def is_sequence(tree: ProcessTree):
    return tree is not None and tree.operator == Operator.SEQUENCE


def is_reverse_sequence(tree: ProcessTree):
    return tree is not None and tree.operator == Operator.REVERSE_SEQUENCE


def is_parallel(tree: ProcessTree):
    return tree is not None and tree.operator == Operator.PARALLEL


def is_xor(tree: ProcessTree):
    return tree is not None and tree.operator == Operator.XOR


def is_loop(tree: ProcessTree):
    return tree is not None and tree.operator == Operator.LOOP


def is_root(tree: ProcessTree):
    return tree.parent is None


def is_leaf(tree: ProcessTree):
    return len(tree.children) == 0 and tree.operator is None


def is_do_child(tree: ProcessTree):
    return (
        tree.parent is not None
        and is_loop(tree.parent)
        and tree.parent.children[0] is tree
    )


def is_redo_child(tree: ProcessTree):
    return (
        tree.parent is not None
        and is_loop(tree.parent)
        and tree.parent.children[1] is tree
    )


def get_nodes_as_set(tree: ProcessTree, nodes=None) -> Set[ProcessTree]:
    nodes = nodes if nodes is not None else set()

    nodes.add(tree)
    for child in tree.children:
        get_nodes_as_set(child, nodes)

    return nodes


def get_reverse_tree(tree: ProcessTree):

    operator = tree.operator

    if tree.operator == Operator.SEQUENCE:
        operator = Operator.REVERSE_SEQUENCE

    if tree.operator == Operator.REVERSE_SEQUENCE:
        operator = Operator.SEQUENCE

    node: ProcessTree = ProcessTree(operator=operator, label=tree.label, position=tree.position)

    for child in tree.children:
        new_child = get_reverse_tree(child)
        new_child.parent = node
        node.children.append(new_child)

    return node
