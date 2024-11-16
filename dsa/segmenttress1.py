"""
SegmentTrees.py

This module provides a comprehensive implementation of Segment Trees, a powerful data structure
used for efficient range queries and updates on arrays. The implementation covers basic to advanced
concepts, ensuring clarity, maintainability, and scalability. It adheres strictly to PEP-8 standards
and includes comprehensive error handling.

Author: generalaimodels-agent
Date: 2024-11-15
"""

from typing import List, Callable, Optional


class SegmentTree:
    """
    A versatile Segment Tree implementation supporting various range queries and updates.
    """

    def __init__(self, data: List[int], function: Callable[[int, int], int], default: int) -> None:
        """
        Initializes the Segment Tree.

        Args:
            data (List[int]): The input array for which the segment tree is built.
            function (Callable[[int, int], int]): The function to be used for range queries
                (e.g., sum, min, max).
            default (int): The default value for non-overlapping segments.

        Raises:
            ValueError: If the input data list is empty.
            TypeError: If the function provided is not callable.
        """
        if not data:
            raise ValueError("Input data list cannot be empty.")
        if not callable(function):
            raise TypeError("Function must be callable.")

        self.n = len(data)
        self.function = function
        self.default = default
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [self.default] * (2 * self.size)
        self._build(data)

    def _build(self, data: List[int]) -> None:
        """
        Builds the segment tree from the input data.

        Args:
            data (List[int]): The input array.
        """
        # Initialize leaves
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        # Build the tree by calculating parents
        for i in range(self.size - 1, 0, -1):
            left_child = self.tree[2 * i]
            right_child = self.tree[2 * i + 1]
            self.tree[i] = self.function(left_child, right_child)

    def update(self, index: int, value: int) -> None:
        """
        Updates the value at a specific index and recalculates the affected segment.

        Args:
            index (int): The index to update (0-based).
            value (int): The new value to be set.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if index < 0 or index >= self.n:
            raise IndexError("Index out of bounds.")
        # Update the leaf node
        pos = self.size + index
        self.tree[pos] = value
        # Update ancestors
        while pos > 1:
            pos //= 2
            left_child = self.tree[2 * pos]
            right_child = self.tree[2 * pos + 1]
            new_val = self.function(left_child, right_child)
            if self.tree[pos] == new_val:
                break  # No further updates required
            self.tree[pos] = new_val

    def query(self, left: int, right: int) -> int:
        """
        Performs a range query on the interval [left, right).

        Args:
            left (int): The starting index of the range (inclusive).
            right (int): The ending index of the range (exclusive).

        Returns:
            int: The result of applying the function over the specified range.

        Raises:
            IndexError: If the query range is out of bounds.
            ValueError: If left is greater than right.
        """
        if left < 0 or right > self.n:
            raise IndexError("Query range out of bounds.")
        if left > right:
            raise ValueError("Left index cannot be greater than right index.")

        result = self.default
        left += self.size
        right += self.size

        while left < right:
            if left % 2:
                result = self.function(result, self.tree[left])
                left += 1
            if right % 2:
                right -= 1
                result = self.function(result, self.tree[right])
            left //= 2
            right //= 2

        return result

    def __str__(self) -> str:
        """
        Returns a string representation of the segment tree.

        Returns:
            str: The string representation.
        """
        levels = []
        level_size = 1
        while level_size < self.size:
            levels.append(self.tree[level_size:2 * level_size])
            level_size <<= 1
        levels.append(self.tree[self.size:2 * self.size])
        return '\n'.join(['Level {}: {}'.format(i, level) for i, level in enumerate(levels)])


def example_usage() -> None:
    """
    Demonstrates the usage of the SegmentTree class with various operations.
    """
    # Example data
    data = [2, 1, 5, 3, 4, 6]
    # Initialize a segment tree for range sum queries
    sum_tree = SegmentTree(data, function=lambda x, y: x + y, default=0)
    print("Initial Segment Tree for Range Sum:")
    print(sum_tree)
    print()

    # Perform a range sum query from index 1 to 5
    sum_query = sum_tree.query(1, 5)
    print(f"Sum of elements from index 1 to 5: {sum_query}")
    print()

    # Update the value at index 3 to 10
    sum_tree.update(3, 10)
    print("Segment Tree after updating index 3 to 10:")
    print(sum_tree)
    print()

    # Perform the same range sum query again
    sum_query = sum_tree.query(1, 5)
    print(f"Sum of elements from index 1 to 5 after update: {sum_query}")
    print()

    # Initialize a segment tree for range minimum queries
    min_tree = SegmentTree(data, function=min, default=float('inf'))
    print("Initial Segment Tree for Range Minimum:")
    print(min_tree)
    print()

    # Perform a range minimum query from index 0 to 4
    min_query = min_tree.query(0, 4)
    print(f"Minimum of elements from index 0 to 4: {min_query}")
    print()

    # Update the value at index 2 to 0
    min_tree.update(2, 0)
    print("Segment Tree after updating index 2 to 0:")
    print(min_tree)
    print()

    # Perform the same range minimum query again
    min_query = min_tree.query(0, 4)
    print(f"Minimum of elements from index 0 to 4 after update: {min_query}")
    print()


if __name__ == "__main__":
    example_usage()