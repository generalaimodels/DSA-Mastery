"""
Fenwick Tree (Binary Indexed Tree) Implementation in Python

This module provides a comprehensive implementation of the Fenwick Tree data structure,
supporting efficient prefix sum queries and single-element updates. The implementation
adheres to PEP-8 standards, utilizes type hints for clarity, and includes robust error
handling to ensure reliability and maintainability.

Author: OpenAI ChatGPT
Date: 2023-10
"""

from typing import List


class FenwickTree:
    """
    A Fenwick Tree (Binary Indexed Tree) for efficiently computing prefix sums and
    updating elements in a list of numbers.

    Attributes:
        size (int): The size of the Fenwick Tree.
        tree (List[int]): Internal representation of the Fenwick Tree.
    """

    def __init__(self, size: int) -> None:
        """
        Initializes a Fenwick Tree with a given size.

        Args:
            size (int): The number of elements in the Fenwick Tree.

        Raises:
            ValueError: If the size is not a positive integer.
        """
        if not isinstance(size, int):
            raise TypeError(f"Size must be an integer, got {type(size).__name__}")
        if size <= 0:
            raise ValueError("Size of Fenwick Tree must be a positive integer.")

        self.size: int = size
        self.tree: List[int] = [0] * (self.size + 1)

    def update(self, index: int, value: int) -> None:
        """
        Updates the Fenwick Tree by adding a value to the element at the specified index.

        Args:
            index (int): 1-based index of the element to update.
            value (int): The value to add to the element.

        Raises:
            TypeError: If index or value is not an integer.
            IndexError: If the index is out of bounds.
        """
        if not isinstance(index, int):
            raise TypeError(f"Index must be an integer, got {type(index).__name__}")
        if not isinstance(value, int):
            raise TypeError(f"Value must be an integer, got {type(value).__name__}")
        if index < 1 or index > self.size:
            raise IndexError(f"Index {index} is out of bounds for Fenwick Tree of size {self.size}.")

        while index <= self.size:
            self.tree[index] += value
            index += index & -index

    def query(self, index: int) -> int:
        """
        Computes the prefix sum from index 1 to the specified index.

        Args:
            index (int): 1-based index up to which the prefix sum is calculated.

        Returns:
            int: The prefix sum from index 1 to the specified index.

        Raises:
            TypeError: If index is not an integer.
            IndexError: If the index is out of bounds.
        """
        if not isinstance(index, int):
            raise TypeError(f"Index must be an integer, got {type(index).__name__}")
        if index < 1 or index > self.size:
            raise IndexError(f"Index {index} is out of bounds for Fenwick Tree of size {self.size}.")

        result: int = 0
        while index > 0:
            result += self.tree[index]
            index -= index & -index
        return result

    def range_query(self, left: int, right: int) -> int:
        """
        Computes the sum of elements in the range [left, right].

        Args:
            left (int): 1-based starting index of the range.
            right (int): 1-based ending index of the range.

        Returns:
            int: The sum of elements from index left to right.

        Raises:
            TypeError: If left or right is not an integer.
            IndexError: If left or right is out of bounds.
            ValueError: If left is greater than right.
        """
        if not all(isinstance(i, int) for i in (left, right)):
            raise TypeError("Left and right indices must be integers.")
        if left < 1 or right > self.size:
            raise IndexError(f"Range [{left}, {right}] is out of bounds for Fenwick Tree of size {self.size}.")
        if left > right:
            raise ValueError(f"Left index {left} cannot be greater than right index {right}.")

        return self.query(right) - self.query(left - 1)

    def build(self, data: List[int]) -> None:
        """
        Builds the Fenwick Tree from a given list of integers.

        Args:
            data (List[int]): The list of integers to build the Fenwick Tree from.

        Raises:
            TypeError: If data is not a list of integers.
            ValueError: If the length of data does not match the size of the Fenwick Tree.
        """
        if not isinstance(data, list):
            raise TypeError(f"Data must be a list, got {type(data).__name__}")
        if not all(isinstance(x, int) for x in data):
            raise TypeError("All elements in data must be integers.")
        if len(data) != self.size:
            raise ValueError(f"Data length {len(data)} does not match Fenwick Tree size {self.size}.")

        for idx, value in enumerate(data, start=1):
            self.update(idx, value)

    def __repr__(self) -> str:
        """
        Returns the official string representation of the Fenwick Tree.

        Returns:
            str: The string representation.
        """
        return f"FenwickTree(size={self.size}, tree={self.tree})"


def main() -> None:
    """
    Demonstrates the usage of the FenwickTree class with sample operations.
    """
    try:
        # Initialize Fenwick Tree with size 10
        ft = FenwickTree(10)

        # Build the tree with initial data
        initial_data = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 0th index is dummy
        ft.build(initial_data)

        print("Initial Fenwick Tree:", ft)

        # Update index 5 by adding 10
        ft.update(5, 10)
        print("After updating index 5 by 10:", ft)

        # Query prefix sum up to index 5
        prefix_sum = ft.query(5)
        print(f"Prefix sum up to index 5: {prefix_sum}")

        # Query range sum from index 3 to 7
        range_sum = ft.range_query(3, 7)
        print(f"Range sum from index 3 to 7: {range_sum}")

    except (TypeError, ValueError, IndexError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()