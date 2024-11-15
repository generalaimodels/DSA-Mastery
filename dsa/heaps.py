"""
Heaps Implementation in Python

This module provides a comprehensive implementation of the Heap data structure,
covering basic to advanced concepts. It adheres to PEP-8 standards, utilizes
type hints for clarity, and includes robust error handling to ensure reliability
and scalability.

Author: OpenAI ChatGPT
Date: 2023-10
"""

from typing import List, TypeVar, Generic, Optional
import torch


T = TypeVar('T')


class HeapEmptyError(Exception):
    """Custom exception to indicate that the heap is empty."""
    pass


class Heap(Generic[T]):
    """
    A generic Heap implementation supporting both Min-Heap and Max-Heap.

    This class provides methods to insert elements, extract the root element,
    peek at the root, and build a heap from an existing list. It supports
    both Min-Heap and Max-Heap configurations based on the comparator provided.
    """

    def __init__(self, comparator: Optional[callable] = None) -> None:
        """
        Initialize a new Heap instance.

        Args:
            comparator (Optional[callable]): A function that compares two elements.
                For a Min-Heap, use the default comparator (lambda x, y: x < y).
                For a Max-Heap, pass a comparator (lambda x, y: x > y).
                If None, defaults to Min-Heap behavior.
        """
        self.heap: List[T] = []
        if comparator is None:
            # Default comparator for Min-Heap
            self.comparator = lambda x, y: x < y
        else:
            self.comparator = comparator

    def _parent(self, index: int) -> int:
        """Return the parent index of the given node index."""
        return (index - 1) // 2

    def _left_child(self, index: int) -> int:
        """Return the left child index of the given node index."""
        return 2 * index + 1

    def _right_child(self, index: int) -> int:
        """Return the right child index of the given node index."""
        return 2 * index + 2

    def _has_left(self, index: int) -> bool:
        """Check if the node at index has a left child."""
        return self._left_child(index) < len(self.heap)

    def _has_right(self, index: int) -> bool:
        """Check if the node at index has a right child."""
        return self._right_child(index) < len(self.heap)

    def _swap(self, i: int, j: int) -> None:
        """Swap two nodes in the heap."""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _heapify_up(self, index: int) -> None:
        """
        Move the node at the given index up to its correct position to maintain the heap property.

        Args:
            index (int): The index of the node to heapify upwards.
        """
        while index > 0:
            parent_index = self._parent(index)
            if self.comparator(self.heap[index], self.heap[parent_index]):
                self._swap(index, parent_index)
                index = parent_index
            else:
                break

    def _heapify_down(self, index: int) -> None:
        """
        Move the node at the given index down to its correct position to maintain the heap property.

        Args:
            index (int): The index of the node to heapify downwards.
        """
        size = len(self.heap)
        while self._has_left(index):
            larger_child_index = self._left_child(index)
            if self._has_right(index):
                if self.comparator(self.heap[self._right_child(index)], self.heap[larger_child_index]):
                    larger_child_index = self._right_child(index)
            if self.comparator(self.heap[larger_child_index], self.heap[index]):
                self._swap(index, larger_child_index)
                index = larger_child_index
            else:
                break

    def insert(self, element: T) -> None:
        """
        Insert a new element into the heap.

        Args:
            element (T): The element to be inserted.
        """
        self.heap.append(element)
        self._heapify_up(len(self.heap) - 1)

    def extract_root(self) -> T:
        """
        Remove and return the root element of the heap.

        Returns:
            T: The root element of the heap.

        Raises:
            HeapEmptyError: If the heap is empty.
        """
        if not self.heap:
            raise HeapEmptyError("Cannot extract from an empty heap.")
        root = self.heap[0]
        last_element = self.heap.pop()
        if self.heap:
            self.heap[0] = last_element
            self._heapify_down(0)
        return root

    def peek(self) -> T:
        """
        Return the root element of the heap without removing it.

        Returns:
            T: The root element of the heap.

        Raises:
            HeapEmptyError: If the heap is empty.
        """
        if not self.heap:
            raise HeapEmptyError("Heap is empty.")
        return self.heap[0]

    def build_heap(self, elements: List[T]) -> None:
        """
        Build a heap from an existing list of elements.

        Args:
            elements (List[T]): The list of elements to build the heap from.
        """
        self.heap = elements[:]
        for i in reversed(range(len(self.heap) // 2)):
            self._heapify_down(i)

    def size(self) -> int:
        """
        Get the number of elements in the heap.

        Returns:
            int: The size of the heap.
        """
        return len(self.heap)

    def is_empty(self) -> bool:
        """
        Check if the heap is empty.

        Returns:
            bool: True if the heap is empty, False otherwise.
        """
        return len(self.heap) == 0

    def __str__(self) -> str:
        """
        Return a string representation of the heap.

        Returns:
            str: The string representation of the heap.
        """
        return str(self.heap)

    def heap_sort(self, ascending: bool = True) -> List[T]:
        """
        Perform heap sort on the elements of the heap.

        Args:
            ascending (bool): If True, sort in ascending order; otherwise, sort in descending order.

        Returns:
            List[T]: A new list containing the sorted elements.
        """
        original_heap = self.heap.copy()
        sorted_list = []
        try:
            while not self.is_empty():
                root = self.extract_root()
                sorted_list.append(root)
        except HeapEmptyError:
            pass

        # Restore the original heap
        self.heap = original_heap[:]
        if ascending:
            sorted_list.reverse()
        return sorted_list

    def merge_heaps(self, other_heap: 'Heap[T]') -> None:
        """
        Merge another heap into this heap.

        Args:
            other_heap (Heap[T]): The other heap to merge.
        """
        self.heap.extend(other_heap.heap)
        for i in reversed(range(len(self.heap) // 2)):
            self._heapify_down(i)


def main() -> None:
    """
    Demonstrate the usage of the Heap class with various operations.
    """
    print("=== Heap Demonstration ===\n")

    # Create a Min-Heap
    min_heap = Heap[int]()
    elements = [5, 3, 8, 1, 2, 9]
    print(f"Building Min-Heap with elements: {elements}")
    min_heap.build_heap(elements)
    print(f"Min-Heap: {min_heap}")

    # Insert an element
    print("\nInserting element 0 into Min-Heap.")
    min_heap.insert(0)
    print(f"Min-Heap after insertion: {min_heap}")

    # Extract the root
    print("\nExtracting the root from Min-Heap.")
    root = min_heap.extract_root()
    print(f"Extracted root: {root}")
    print(f"Min-Heap after extraction: {min_heap}")

    # Peek at the root
    print("\nPeeking at the root of Min-Heap.")
    root_peek = min_heap.peek()
    print(f"Root element: {root_peek}")

    # Perform heap sort
    print("\nPerforming heap sort (ascending) on Min-Heap.")
    sorted_elements = min_heap.heap_sort(ascending=True)
    print(f"Sorted elements: {sorted_elements}")

    # Create a Max-Heap
    max_heap = Heap[int](comparator=lambda x, y: x > y)
    elements_max = [5, 3, 8, 1, 2, 9]
    print(f"\nBuilding Max-Heap with elements: {elements_max}")
    max_heap.build_heap(elements_max)
    print(f"Max-Heap: {max_heap}")

    # Insert an element
    print("\nInserting element 10 into Max-Heap.")
    max_heap.insert(10)
    print(f"Max-Heap after insertion: {max_heap}")

    # Extract the root
    print("\nExtracting the root from Max-Heap.")
    max_root = max_heap.extract_root()
    print(f"Extracted root: {max_root}")
    print(f"Max-Heap after extraction: {max_heap}")

    # Perform heap sort
    print("\nPerforming heap sort (descending) on Max-Heap.")
    sorted_max_elements = max_heap.heap_sort(ascending=False)
    print(f"Sorted elements: {sorted_max_elements}")

    # Demonstrate error handling
    print("\nAttempting to extract from an empty heap.")
    empty_heap = Heap[int]()
    try:
        empty_heap.extract_root()
    except HeapEmptyError as e:
        print(f"Caught an error: {e}")

    print("\n=== End of Demonstration ===")


if __name__ == "__main__":
    main()