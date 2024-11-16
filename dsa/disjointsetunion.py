"""
Disjoint Set Union (DSU) Implementation and Advanced Topics
============================================================

This module provides a comprehensive implementation of the Disjoint Set Union (DSU) data structure,
also known as Union-Find. It covers basic to advanced concepts, ensuring adherence to PEP-8 standards,
utilizing Python's typing module for clarity, and focusing on performance, scalability, and robustness.

Author: generalaimodels-agent
Date: 2024-11-15
"""

from typing import Any, Dict, List, Optional


class DSU:
    """
    Disjoint Set Union (DSU) or Union-Find data structure implementation.

    DSU keeps track of a set of elements partitioned into a number of disjoint (non-overlapping) subsets.
    It provides near-constant-time operations to add new sets, merge existing sets, and determine whether
    elements are in the same set.

    Attributes:
        parent (Dict[Any, Any]): A dictionary mapping each element to its parent.
        rank (Dict[Any, int]): A dictionary mapping each element to its rank for union by rank optimization.
    """

    def __init__(self) -> None:
        """
        Initializes an empty DSU.
        """
        self.parent: Dict[Any, Any] = {}
        self.rank: Dict[Any, int] = {}

    def make_set(self, x: Any) -> None:
        """
        Creates a new set containing the single element x.

        Args:
            x (Any): The element to be added as a new set.

        Raises:
            ValueError: If the element x already exists in the DSU.
        """
        if x in self.parent:
            raise ValueError(f"Element {x} already exists.")
        self.parent[x] = x
        self.rank[x] = 0
        print(f"Created set: {x}")

    def find(self, x: Any) -> Any:
        """
        Finds the representative (root) of the set containing x with path compression.

        Args:
            x (Any): The element to find.

        Returns:
            Any: The representative of the set containing x.

        Raises:
            KeyError: If the element x is not present in the DSU.
        """
        if x not in self.parent:
            raise KeyError(f"Element {x} not found.")
        if self.parent[x] != x:
            print(f"Path compression for {x}: {self.parent[x]} -> find({self.parent[x]})")
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: Any, y: Any) -> None:
        """
        Unites the sets containing x and y using union by rank.

        Args:
            x (Any): The first element.
            y (Any): The second element.

        Raises:
            KeyError: If either x or y is not present in the DSU.
        """
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            print(f"Elements {x} and {y} are already in the same set.")
            return

        # Union by rank optimization
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
            print(f"Union by rank: {x_root} -> {y_root}")
        else:
            self.parent[y_root] = x_root
            print(f"Union by rank: {y_root} -> {x_root}")
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1
                print(f"Incremented rank of {x_root} to {self.rank[x_root]}")

    def connected(self, x: Any, y: Any) -> bool:
        """
        Determines whether elements x and y are in the same set.

        Args:
            x (Any): The first element.
            y (Any): The second element.

        Returns:
            bool: True if x and y are in the same set, False otherwise.

        Raises:
            KeyError: If either x or y is not present in the DSU.
        """
        return self.find(x) == self.find(y)

    def get_sets(self) -> Dict[Any, List[Any]]:
        """
        Retrieves all sets as a dictionary mapping set representatives to their members.

        Returns:
            Dict[Any, List[Any]]: A dictionary of all sets.
        """
        sets: Dict[Any, List[Any]] = {}
        for element in self.parent:
            root = self.find(element)
            if root in sets:
                sets[root].append(element)
            else:
                sets[root] = [element]
        return sets

    def size(self, x: Any) -> int:
        """
        Returns the size of the set containing element x.

        Args:
            x (Any): The element to query.

        Returns:
            int: The size of the set containing x.

        Raises:
            KeyError: If the element x is not present in the DSU.
        """
        root = self.find(x)
        return len(self.get_sets()[root])

    def __str__(self) -> str:
        """
        Returns a string representation of the DSU.

        Returns:
            str: String representation of all sets.
        """
        sets = self.get_sets()
        set_strings = [f"{root}: {members}" for root, members in sets.items()]
        return "{" + ", ".join(set_strings) + "}"

    def __len__(self) -> int:
        """
        Returns the number of disjoint sets.

        Returns:
            int: Number of disjoint sets.
        """
        return len(self.get_sets())

    def reset(self) -> None:
        """
        Resets the DSU to an empty state.
        """
        self.parent.clear()
        self.rank.clear()
        print("DSU has been reset.")


def main() -> None:
    """
    Demonstrates the usage of the DSU class with various operations.
    """
    dsu = DSU()
    
    # Example elements
    elements = ['A', 'B', 'C', 'D', 'E', 'F']
    
    # Create sets
    for elem in elements:
        dsu.make_set(elem)
    
    print("\nInitial sets:")
    print(dsu)
    
    # Perform unions
    dsu.union('A', 'B')
    dsu.union('C', 'D')
    dsu.union('E', 'F')
    dsu.union('B', 'C')
    
    print("\nSets after unions:")
    print(dsu)
    
    # Check connectivity
    print(f"\nAre A and D connected? {dsu.connected('A', 'D')}")
    print(f"Are A and E connected? {dsu.connected('A', 'E')}")
    
    # Get size of a set
    print(f"\nSize of set containing A: {dsu.size('A')}")
    print(f"Size of set containing E: {dsu.size('E')}")
    
    # Get all sets
    print("\nAll disjoint sets:")
    all_sets = dsu.get_sets()
    for root, members in all_sets.items():
        print(f"Set {root}: {members}")
    
    # Reset DSU
    dsu.reset()
    print("\nAfter reset:")
    print(dsu)


if __name__ == "__main__":
    main()