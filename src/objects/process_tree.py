from enum import Enum
from typing import List
from PrettyPrint import PrettyPrintTree


class Operator(Enum):
    SEQUENCE = "->"
    XOR = "X"
    PARALLEL = "+"
    LOOP = "*"
    REVERSE_SEQUENCE = "<-"


class ProcessTree:
    def __init__(
        self, operator: Operator = None, label: str = None, parent: "ProcessTree" = None
    ):
        self._operator = operator
        self._label = label
        self._parent = parent
        self._children: List["ProcessTree"] = []

    @property
    def parent(self) -> "ProcessTree":
        return self._parent

    @parent.setter
    def parent(self, parent: "ProcessTree"):
        self._parent = parent

    @property
    def children(self) -> List["ProcessTree"]:
        return self._children

    @property
    def operator(self) -> Operator:
        return self._operator

    @property
    def label(self) -> str:
        return self._label

    def print_tree(self):
        printer = PrettyPrintTree(
            lambda tree: tree.children,
            lambda tree: (
                tree.operator.value if tree.operator is not None else tree.label
            ),
        )
        printer(self)

    def __repr__(self):
        if self.operator:
            return f"{self.operator.value}({','.join(str(child) for child in self.children)})"
        return f"'{self.label}'"
