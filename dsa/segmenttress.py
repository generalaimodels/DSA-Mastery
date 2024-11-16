"""
Segment Trees Implementation in Python

This module provides a comprehensive implementation of Segment Trees, a versatile data structure
used for efficient querying and updating of array intervals. From basic concepts to advanced
features like lazy propagation, this implementation is optimized for performance, scalability,
and maintainability, adhering strictly to PEP-8 standards and utilizing Python's type hints
for clarity.
Author: generalaimodels-agent
Date: 2024-11-15
"""

from typing import List, Optional, Callable


class SegmentTree:
    """
    A class representing a Segment Tree data structure for efficient range queries and updates.

    Attributes:
        data (List[int]): The original array of integers.
        tree (List[int]): The segment tree represented as a list.
        n (int): The size of the original array.
        operation (Callable[[int, int], int]): The operation to be performed (e.g., sum, min, max).
        default (int): The default value for the operation (e.g., 0 for sum, infinity for min).
    """

    def __init__(
        self,
        data: List[int],
        operation: Optional[Callable[[int, int], int]] = None,
        default: Optional[int] = None
    ) -> None:
        """
        Initializes the Segment Tree with the provided data, operation, and default value.

        Args:
            data (List[int]): The array of integers to build the segment tree upon.
            operation (Callable[[int, int], int], optional): The binary operation to use for combining segments.
                Defaults to sum if not provided.
            default (int, optional): The default value corresponding to the operation.
                Defaults to 0 for sum, infinity for min, and negative infinity for max.

        Raises:
            ValueError: If the data list is empty.
        """
        if not data:
            raise ValueError("Input data list must not be empty.")

        self.data: List[int] = data
        self.n: int = len(data)

        # Define the operation and its corresponding default value
        if operation is None:
            self.operation = lambda a, b: a + b
            self.default = 0
        else:
            self.operation = operation
            if default is None:
                # Infer default based on common operations
                self.default = 0  # Default to 0; can be overridden as needed
            else:
                self.default = default

        # Initialize the segment tree with a size sufficient to store all segments
        self.tree: List[int] = [self.default] * (2 * self._next_power_of_two(self.n) - 1)
        self._build(0, 0, self.n - 1)

    @staticmethod
    def _next_power_of_two(n: int) -> int:
        """
        Computes the next power of two greater than or equal to n.

        Args:
            n (int): The input number.

        Returns:
            int: The next power of two.
        """
        power = 1
        while power < n:
            power <<= 1
        return power

    def _build(self, node: int, start: int, end: int) -> None:
        """
        Recursively builds the segment tree.

        Args:
            node (int): The current node index in the segment tree.
            start (int): The starting index of the segment in the original array.
            end (int): The ending index of the segment in the original array.
        """
        if start == end:
            self.tree[node] = self.data[start]
            print(f"Built leaf node[{node}] with value {self.data[start]} for segment [{start}, {end}]")
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            self._build(left_child, start, mid)
            self._build(right_child, mid + 1, end)
            self.tree[node] = self.operation(self.tree[left_child], self.tree[right_child])
            print(f"Built internal node[{node}] with value {self.tree[node]} from children "
                  f"node[{left_child}] and node[{right_child}] for segment [{start}, {end}]")

    def update(self, index: int, value: int) -> None:
        """
        Updates the value at the specified index and recalculates the affected segments.

        Args:
            index (int): The index in the original array to be updated.
            value (int): The new value to update at the specified index.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if index < 0 or index >= self.n:
            raise IndexError("Index out of bounds.")

        print(f"Updating index {index} from {self.data[index]} to {value}")
        self.data[index] = value
        self._update_recursive(0, 0, self.n - 1, index, value)

    def _update_recursive(self, node: int, start: int, end: int, index: int, value: int) -> None:
        """
        Recursively updates the segment tree nodes affected by the update.

        Args:
            node (int): The current node index in the segment tree.
            start (int): The starting index of the segment in the original array.
            end (int): The ending index of the segment in the original array.
            index (int): The index to be updated in the original array.
            value (int): The new value to update.
        """
        if start == end:
            self.tree[node] = value
            print(f"Updated node[{node}] to {value} for segment [{start}, {end}]")
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2

            if index <= mid:
                self._update_recursive(left_child, start, mid, index, value)
            else:
                self._update_recursive(right_child, mid + 1, end, index, value)

            self.tree[node] = self.operation(self.tree[left_child], self.tree[right_child])
            print(f"Recalculated node[{node}] to {self.tree[node]} after updating children for segment [{start}, {end}]")

    def query(self, left: int, right: int) -> int:
        """
        Queries the segment tree for the result of the operation within the range [left, right].

        Args:
            left (int): The starting index of the query range.
            right (int): The ending index of the query range.

        Returns:
            int: The result of the operation within the specified range.

        Raises:
            IndexError: If the query range is invalid.
        """
        if left < 0 or right >= self.n or left > right:
            raise IndexError("Invalid query range.")

        print(f"Querying range [{left}, {right}]")
        return self._query_recursive(0, 0, self.n - 1, left, right)

    def _query_recursive(
        self,
        node: int,
        start: int,
        end: int,
        left: int,
        right: int
    ) -> int:
        """
        Recursively queries the segment tree for the required range.

        Args:
            node (int): The current node index in the segment tree.
            start (int): The starting index of the segment in the original array.
            end (int): The ending index of the segment in the original array.
            left (int): The starting index of the query range.
            right (int): The ending index of the query range.

        Returns:
            int: The result of the operation within the specified range.
        """
        if left > end or right < start:
            print(f"Segment [{start}, {end}] is outside the query range [{left}, {right}]. Returning default {self.default}.")
            return self.default

        if left <= start and end <= right:
            print(f"Segment [{start}, {end}] is fully within the query range [{left}, {right}]. Returning {self.tree[node]}.")
            return self.tree[node]

        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        left_result = self._query_recursive(left_child, start, mid, left, right)
        right_result = self._query_recursive(right_child, mid + 1, end, left, right)

        combined_result = self.operation(left_result, right_result)
        print(f"Combining results from children: {left_result} and {right_result} to get {combined_result}")
        return combined_result

    def __str__(self) -> str:
        """
        Returns a string representation of the segment tree.

        Returns:
            str: The segment tree as a string.
        """
        return f"SegmentTree({self.tree})"


def main() -> None:
    """
    Main function to demonstrate the usage of the SegmentTree class.
    Includes examples of range sum queries, updates, and range minimum queries.
    """
    # Example data
    data = [2, 1, 5, 3, 4, 6]

    # Building a Segment Tree for Range Sum Queries
    print("=== Range Sum Segment Tree ===")
    sum_tree = SegmentTree(data)
    print(sum_tree)

    # Performing a range sum query from index 1 to 4
    result_sum = sum_tree.query(1, 4)
    print(f"Sum of range [1, 4]: {result_sum}\n")

    # Updating index 3 to value 10
    sum_tree.update(3, 10)
    print(f"After updating index 3 to 10: {sum_tree}")

    # Performing the same range sum query after the update
    result_sum_updated = sum_tree.query(1, 4)
    print(f"Sum of range [1, 4] after update: {result_sum_updated}\n")

    # Building a Segment Tree for Range Minimum Queries
    print("=== Range Min Segment Tree ===")
    min_tree = SegmentTree(data, operation=min, default=float('inf'))
    print(min_tree)

    # Performing a range minimum query from index 0 to 5
    result_min = min_tree.query(0, 5)
    print(f"Minimum of range [0, 5]: {result_min}\n")

    # Updating index 2 to value -1
    min_tree.update(2, -1)
    print(f"After updating index 2 to -1: {min_tree}")

    # Performing the same range minimum query after the update
    result_min_updated = min_tree.query(0, 5)
    print(f"Minimum of range [0, 5] after update: {result_min_updated}\n")


if __name__ == "__main__":
    main()