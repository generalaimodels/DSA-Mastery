"""
Heap Data Structure Module
==========================

This module provides a comprehensive implementation of the Heap data structure in Python,
covering basic to advanced topics. It includes both MinHeap and MaxHeap classes, supporting
common operations such as insertion, deletion, heapify, and heap sort. The implementation
adheres to PEP-8 standards, utilizes type hints for clarity, and includes comprehensive
error handling to ensure robustness and scalability.

Author: OpenAI ChatGPT
Date: 2023-10
"""

from typing import Any, List, Optional, TypeVar, Generic

T = TypeVar('T')


class HeapError(Exception):
    """Custom exception class for Heap-related errors."""
    pass


class Heap(Generic[T]):
    """
    A generic Heap class that can be used as a base for MinHeap and MaxHeap.

    Attributes:
        heap_list (List[T]): The list that holds the heap elements.
        heap_size (int): The current number of elements in the heap.
    """

    def __init__(self) -> None:
        """Initialize an empty heap."""
        self.heap_list: List[T] = []
        self.heap_size: int = 0

    def parent(self, index: int) -> int:
        """Return the index of the parent of the given node."""
        return (index - 1) // 2

    def left_child(self, index: int) -> int:
        """Return the index of the left child of the given node."""
        return 2 * index + 1

    def right_child(self, index: int) -> int:
        """Return the index of the right child of the given node."""
        return 2 * index + 2

    def has_parent(self, index: int) -> bool:
        """Check if the node at the given index has a parent."""
        return self.parent(index) >= 0

    def has_left_child(self, index: int) -> bool:
        """Check if the node at the given index has a left child."""
        return self.left_child(index) < self.heap_size

    def has_right_child(self, index: int) -> bool:
        """Check if the node at the given index has a right child."""
        return self.right_child(index) < self.heap_size

    def heapify_up(self, index: int) -> None:
        """
        Restore the heap property moving upwards from the given index.

        Args:
            index (int): The index to start heapifying up from.
        """
        raise NotImplementedError("heapify_up must be implemented by subclasses.")

    def heapify_down(self, index: int) -> None:
        """
        Restore the heap property moving downwards from the given index.

        Args:
            index (int): The index to start heapifying down from.
        """
        raise NotImplementedError("heapify_down must be implemented by subclasses.")

    def insert(self, item: T) -> None:
        """
        Insert a new item into the heap.

        Args:
            item (T): The item to be inserted.
        """
        self.heap_list.append(item)
        self.heap_size += 1
        self.heapify_up(self.heap_size - 1)

    def peek(self) -> T:
        """
        Get the top element of the heap without removing it.

        Returns:
            T: The top element of the heap.

        Raises:
            HeapError: If the heap is empty.
        """
        if self.heap_size == 0:
            raise HeapError("Heap is empty.")
        return self.heap_list[0]

    def extract_top(self) -> T:
        """
        Remove and return the top element of the heap.

        Returns:
            T: The top element of the heap.

        Raises:
            HeapError: If the heap is empty.
        """
        if self.heap_size == 0:
            raise HeapError("Heap is empty.")
        top_item = self.heap_list[0]
        last_item = self.heap_list.pop()
        self.heap_size -= 1
        if self.heap_size > 0:
            self.heap_list[0] = last_item
            self.heapify_down(0)
        return top_item

    def build_heap(self, items: List[T]) -> None:
        """
        Build a heap from a list of items.

        Args:
            items (List[T]): The list of items to build the heap from.
        """
        self.heap_list = items.copy()
        self.heap_size = len(self.heap_list)
        for index in reversed(range(self.heap_size // 2)):
            self.heapify_down(index)

    def is_empty(self) -> bool:
        """
        Check if the heap is empty.

        Returns:
            bool: True if the heap is empty, False otherwise.
        """
        return self.heap_size == 0

    def heap_sort(self, ascending: bool = True) -> List[T]:
        """
        Perform heap sort on the heap elements.

        Args:
            ascending (bool): If True, sort in ascending order; otherwise, descending.

        Returns:
            List[T]: The sorted list of elements.
        """
        sorted_list: List[T] = []
        cloned_heap = self.clone()

        while not cloned_heap.is_empty():
            sorted_list.append(cloned_heap.extract_top())

        if ascending:
            sorted_list.reverse()
        return sorted_list

    def clone(self) -> 'Heap[T]':
        """
        Create a clone of the heap.

        Returns:
            Heap[T]: A new heap with the same elements.
        """
        raise NotImplementedError("clone must be implemented by subclasses.")

    def __len__(self) -> int:
        """Return the number of elements in the heap."""
        return self.heap_size

    def __str__(self) -> str:
        """Return the string representation of the heap."""
        return str(self.heap_list)


class MinHeap(Heap[T]):
    """
    A MinHeap implementation where the smallest element is at the top.
    """

    def heapify_up(self, index: int) -> None:
        """Restore the min-heap property moving upwards from the given index."""
        current = index
        while self.has_parent(current):
            parent_idx = self.parent(current)
            if self.heap_list[current] < self.heap_list[parent_idx]:
                self.heap_list[current], self.heap_list[parent_idx] = (
                    self.heap_list[parent_idx],
                    self.heap_list[current],
                )
                current = parent_idx
            else:
                break

    def heapify_down(self, index: int) -> None:
        """Restore the min-heap property moving downwards from the given index."""
        current = index
        while self.has_left_child(current):
            smaller_child_idx = self.left_child(current)
            if (
                self.has_right_child(current)
                and self.heap_list[self.right_child(current)] < self.heap_list[smaller_child_idx]
            ):
                smaller_child_idx = self.right_child(current)

            if self.heap_list[current] > self.heap_list[smaller_child_idx]:
                self.heap_list[current], self.heap_list[smaller_child_idx] = (
                    self.heap_list[smaller_child_idx],
                    self.heap_list[current],
                )
                current = smaller_child_idx
            else:
                break

    def clone(self) -> 'MinHeap[T]':
        """Create a clone of the MinHeap."""
        cloned = MinHeap[T]()
        cloned.heap_list = self.heap_list.copy()
        cloned.heap_size = self.heap_size
        return cloned


class MaxHeap(Heap[T]):
    """
    A MaxHeap implementation where the largest element is at the top.
    """

    def heapify_up(self, index: int) -> None:
        """Restore the max-heap property moving upwards from the given index."""
        current = index
        while self.has_parent(current):
            parent_idx = self.parent(current)
            if self.heap_list[current] > self.heap_list[parent_idx]:
                self.heap_list[current], self.heap_list[parent_idx] = (
                    self.heap_list[parent_idx],
                    self.heap_list[current],
                )
                current = parent_idx
            else:
                break

    def heapify_down(self, index: int) -> None:
        """Restore the max-heap property moving downwards from the given index."""
        current = index
        while self.has_left_child(current):
            larger_child_idx = self.left_child(current)
            if (
                self.has_right_child(current)
                and self.heap_list[self.right_child(current)] > self.heap_list[larger_child_idx]
            ):
                larger_child_idx = self.right_child(current)

            if self.heap_list[current] < self.heap_list[larger_child_idx]:
                self.heap_list[current], self.heap_list[larger_child_idx] = (
                    self.heap_list[larger_child_idx],
                    self.heap_list[current],
                )
                current = larger_child_idx
            else:
                break

    def clone(self) -> 'MaxHeap[T]':
        """Create a clone of the MaxHeap."""
        cloned = MaxHeap[T]()
        cloned.heap_list = self.heap_list.copy()
        cloned.heap_size = self.heap_size
        return cloned


def heap_sort(items: List[T], ascending: bool = True) -> List[T]:
    """
    Sort a list of items using heap sort.

    Args:
        items (List[T]): The list of items to sort.
        ascending (bool): If True, sort in ascending order; otherwise, descending.

    Returns:
        List[T]: The sorted list of elements.
    """
    if ascending:
        heap = MinHeap[T]()
    else:
        heap = MaxHeap[T]()
    heap.build_heap(items)
    return heap.heap_sort(ascending=ascending)


def main() -> None:
    """
    Demonstrate the usage of MinHeap and MaxHeap with sample data.
    """
    try:
        # Sample data
        data = [15, 10, 20, 17, 25]

        print("=== MinHeap Demonstration ===")
        min_heap = MinHeap[int]()
        for item in data:
            min_heap.insert(item)
            print(f"Inserted {item}: {min_heap}")

        print(f"Top element: {min_heap.peek()}")
        print(f"Extracted top element: {min_heap.extract_top()}")
        print(f"Heap after extraction: {min_heap}")

        print("\nHeap Sort (Ascending):")
        sorted_asc = heap_sort(data, ascending=True)
        print(sorted_asc)

        print("\n=== MaxHeap Demonstration ===")
        max_heap = MaxHeap[int]()
        for item in data:
            max_heap.insert(item)
            print(f"Inserted {item}: {max_heap}")

        print(f"Top element: {max_heap.peek()}")
        print(f"Extracted top element: {max_heap.extract_top()}")
        print(f"Heap after extraction: {max_heap}")

        print("\nHeap Sort (Descending):")
        sorted_desc = heap_sort(data, ascending=False)
        print(sorted_desc)

    except HeapError as he:
        print(f"Heap error: {he}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()