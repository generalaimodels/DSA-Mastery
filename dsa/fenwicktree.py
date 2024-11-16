"""
Fenwick Tree (Binary Indexed Tree) Implementation in Python

This module provides an efficient implementation of the Fenwick Tree (Binary Indexed Tree)
data structure, which supports prefix sum queries and point updates in logarithmic time.
Suitable for scenarios involving frequent updates and range queries on an array of numbers.

Author: generalaimodels-agent
Date: 2024-11-15
"""

from typing import List, Optional


class FenwickTree:
    """
    A class representing a Fenwick Tree (Binary Indexed Tree) for efficient
    prefix sum queries and point updates.

    Attributes:
        size (int): The size of the Fenwick Tree.
        tree (List[int]): Internal representation of the Fenwick Tree.
    """

    def __init__(self, size: int) -> None:
        """
        Initializes a Fenwick Tree with a specified size.

        Args:
            size (int): The number of elements in the Fenwick Tree.

        Raises:
            ValueError: If the provided size is not a positive integer.
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Size must be a positive integer.")
        self.size: int = size
        self.tree: List[int] = [0] * (self.size + 1)

    def _lowbit(self, index: int) -> int:
        """
        Computes the lowest bit of the given index.

        Args:
            index (int): The index to compute the low bit for.

        Returns:
            int: The lowest bit of the index.
        """
        return index & -index

    def update(self, index: int, delta: int) -> None:
        """
        Updates the Fenwick Tree by adding 'delta' to the element at the given index.

        Args:
            index (int): The 1-based index to update.
            delta (int): The value to add to the element.

        Raises:
            IndexError: If the index is out of bounds.
            TypeError: If 'delta' is not an integer.
        """
        if not isinstance(index, int):
            raise TypeError("Index must be an integer.")
        if not isinstance(delta, int):
            raise TypeError("Delta must be an integer.")
        if index <= 0 or index > self.size:
            raise IndexError("Index out of bounds.")

        while index <= self.size:
            self.tree[index] += delta
            index += self._lowbit(index)

    def query(self, index: int) -> int:
        """
        Computes the prefix sum from index 1 to the given index.

        Args:
            index (int): The 1-based index up to which the sum is computed.

        Returns:
            int: The prefix sum from index 1 to the given index.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if not isinstance(index, int):
            raise TypeError("Index must be an integer.")
        if index < 0 or index > self.size:
            raise IndexError("Index out of bounds.")

        result = 0
        while index > 0:
            result += self.tree[index]
            index -= self._lowbit(index)
        return result

    def range_query(self, left: int, right: int) -> int:
        """
        Computes the sum of elements from index 'left' to 'right'.

        Args:
            left (int): The starting 1-based index.
            right (int): The ending 1-based index.

        Returns:
            int: The sum of elements from left to right.

        Raises:
            ValueError: If 'left' is greater than 'right'.
            IndexError: If either index is out of bounds.
        """
        if left > right:
            raise ValueError("Left index cannot be greater than right index.")
        return self.query(right) - self.query(left - 1)

    def build(self, nums: Optional[List[int]] = None) -> None:
        """
        Builds the Fenwick Tree from an initial list of numbers.

        Args:
            nums (Optional[List[int]]): The initial list of numbers.

        Raises:
            ValueError: If the length of 'nums' does not match the size of the tree.
            TypeError: If 'nums' contains non-integer elements.
        """
        if nums is None:
            return
        if len(nums) != self.size:
            raise ValueError("Length of nums must be equal to the size of the Fenwick Tree.")
        for idx, num in enumerate(nums, start=1):
            if not isinstance(num, int):
                raise TypeError("All elements in nums must be integers.")
            self.update(idx, num)

    def __str__(self) -> str:
        """
        Returns a string representation of the Fenwick Tree.

        Returns:
            str: The string representation.
        """
        return f"FenwickTree(size={self.size}, tree={self.tree})"

    def __repr__(self) -> str:
        """
        Returns the official string representation of the Fenwick Tree.

        Returns:
            str: The official string representation.
        """
        return self.__str__()


def main() -> None:
    """
    Demonstrates the usage of the FenwickTree class with example operations.
    """
    try:
        # Initialize Fenwick Tree with size 10
        size = 10
        fenwick = FenwickTree(size)

        # Build the tree with initial values (without the dummy 0)
        initial_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]  # Length: 10
        fenwick.build(initial_values)

        print("Initial Fenwick Tree:")
        print(fenwick)

        # Update index 5 by adding 10
        fenwick.update(5, 10)
        print("\nAfter updating index 5 by adding 10:")
        print(fenwick)

        # Query prefix sum up to index 5
        prefix_sum = fenwick.query(5)
        print(f"\nPrefix sum up to index 5: {prefix_sum}")

        # Range query from index 3 to 7
        range_sum = fenwick.range_query(3, 7)
        print(f"Range sum from index 3 to 7: {range_sum}")

    except (ValueError, IndexError, TypeError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()