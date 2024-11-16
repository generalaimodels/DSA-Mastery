"""
Disjoint Set Union (DSU) - Comprehensive Implementation and Explanation

DSU, also known as Union-Find, is a fundamental data structure that keeps track of a set of elements
partitioned into a number of disjoint (non-overlapping) subsets. It provides efficient operations
to:

1. **Find**: Determine which subset a particular element is in. This can be used for determining
   if two elements are in the same subset.
2. **Union**: Join two subsets into a single subset.

DSU is widely used in algorithmic applications such as Krusky's Minimum Spanning Tree algorithm,
network connectivity, image processing, and more.

This implementation focuses on optimizing the operations using two techniques:
- **Path Compression**: Flattens the structure of the tree whenever `find` is used, ensuring that each
  node points directly to the root.
- **Union by Rank**: Always attaches the smaller tree to the root of the larger tree to keep the
  tree shallow.

These optimizations ensure that both `find` and `union` operations run in nearly constant time,
specifically O(α(n)), where α is the inverse Ackermann function, which grows extremely slowly.

The following Python implementation adheres to PEP-8 standards, uses type hints from the `typing`
module, includes comprehensive error handling, and is optimized for performance and scalability.

Author: generalaimodels-agent
Date: 2024-11-15
"""

from typing import List


class DSU:
    """
    Disjoint Set Union (Union-Find) data structure with path compression and union by rank.

    Attributes:
        parent (List[int]): The parent representative for each element.
        rank (List[int]): The rank (approximate depth) of each element's tree.
    """

    def __init__(self, size: int) -> None:
        """
        Initializes the DSU with a specified number of elements.

        Args:
            size (int): The number of elements in the DSU.

        Raises:
            ValueError: If the size is not a positive integer.
        """
        if not isinstance(size, int):
            raise TypeError(f"Size must be an integer, got {type(size).__name__}")
        if size <= 0:
            raise ValueError("Size must be a positive integer")

        self.parent: List[int] = list(range(size))
        self.rank: List[int] = [0] * size

    def find(self, x: int) -> int:
        """
        Finds the representative (root) of the set that element `x` belongs to.
        Implements path compression for optimization.

        Args:
            x (int): The element to find.

        Returns:
            int: The representative of the set containing `x`.

        Raises:
            IndexError: If `x` is out of bounds.
            TypeError: If `x` is not an integer.
        """
        if not isinstance(x, int):
            raise TypeError(f"Element must be an integer, got {type(x).__name__}")
        if x < 0 or x >= len(self.parent):
            raise IndexError(f"Element {x} is out of bounds")

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """
        Unites the sets containing elements `x` and `y`.
        Implements union by rank for optimization.

        Args:
            x (int): The first element.
            y (int): The second element.

        Raises:
            IndexError: If `x` or `y` is out of bounds.
            TypeError: If `x` or `y` are not integers.
        """
        if not isinstance(x, int) or not isinstance(y, int):
            raise TypeError("Elements must be integers")
        if x < 0 or x >= len(self.parent) or y < 0 or y >= len(self.parent):
            raise IndexError("One or both elements are out of bounds")

        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return  # Already in the same set

        # Union by rank
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1

    def connected(self, x: int, y: int) -> bool:
        """
        Checks if elements `x` and `y` are in the same set.

        Args:
            x (int): The first element.
            y (int): The second element.

        Returns:
            bool: True if `x` and `y` are in the same set, False otherwise.

        Raises:
            IndexError: If `x` or `y` is out of bounds.
            TypeError: If `x` or `y` are not integers.
        """
        return self.find(x) == self.find(y)

    def get_sets(self) -> List[List[int]]:
        """
        Retrieves all disjoint sets.

        Returns:
            List[List[int]]: A list of sets, each represented as a list of elements.
        """
        sets: dict[int, List[int]] = {}
        for element in range(len(self.parent)):
            root = self.find(element)
            if root in sets:
                sets[root].append(element)
            else:
                sets[root] = [element]
        return list(sets.values())

    def __str__(self) -> str:
        """
        Returns a string representation of the DSU.

        Returns:
            str: String representation showing each element and its parent.
        """
        return f"DSU(parent={self.parent}, rank={self.rank})"

    def __repr__(self) -> str:
        """
        Returns an official string representation of the DSU.

        Returns:
            str: Official string representation.
        """
        return self.__str__()


def main() -> None:
    """
    Demonstrates the usage of the DSU (Disjoint Set Union) data structure.

    This function performs a series of union and find operations and prints the results.
    It includes various test cases to showcase the functionality and error handling
    of the DSU implementation.
    """
    try:
        # Initialize DSU with 10 elements (0 through 9)
        dsu = DSU(10)
        print("Initial DSU:", dsu)

        # Perform some union operations
        dsu.union(0, 1)
        dsu.union(1, 2)
        dsu.union(3, 4)
        dsu.union(5, 6)
        dsu.union(7, 8)
        dsu.union(8, 9)
        print("DSU after unions:", dsu)

        # Check connectivity
        print("Are 0 and 2 connected?", dsu.connected(0, 2))  # True
        print("Are 0 and 3 connected?", dsu.connected(0, 3))  # False
        print("Are 7 and 9 connected?", dsu.connected(7, 9))  # True

        # Union more elements
        dsu.union(2, 3)
        dsu.union(5, 9)
        print("DSU after more unions:", dsu)

        # Check connectivity again
        print("Are 0 and 4 connected?", dsu.connected(0, 4))  # True
        print("Are 5 and 8 connected?", dsu.connected(5, 8))  # True

        # Get all sets
        sets = dsu.get_sets()
        print("Disjoint Sets:", sets)

        # Attempt invalid operations to demonstrate error handling
        try:
            dsu.union(10, 11)  # Out of bounds
        except IndexError as e:
            print("Error:", e)

        try:
            dsu.find("a")  # Invalid type
        except TypeError as e:
            print("Error:", e)

    except Exception as e:
        print("An unexpected error occurred:", e)


if __name__ == "__main__":
    main()