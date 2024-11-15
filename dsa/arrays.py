"""
arrays.py

A comprehensive implementation of array data structures and algorithms in Python,
adhering to PEP-8 standards. This module covers basic to advanced array operations,
including searching, sorting, dynamic resizing, and multi-dimensional arrays.

Dependencies:
    - torch (only if tensor operations are required)

Author: Generalmodelai-agent
Date: 2024-10-15
"""

from typing import TypeVar, Generic, Optional, List, Any

T = TypeVar('T')


class Array(Generic[T]):
    """
    A dynamic array class similar to a simplified Python list.
    """

    def __init__(self, capacity: int = 16) -> None:
        """
        Initializes the array with a specified capacity.

        Args:
            capacity (int): The initial capacity of the array.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive.")
        self._capacity: int = capacity
        self._size: int = 0
        self._data: List[Optional[T]] = [None] * self._capacity

    def __len__(self) -> int:
        """Returns the number of elements in the array."""
        return self._size

    def __getitem__(self, index: int) -> T:
        """
        Retrieves the element at the specified index.

        Args:
            index (int): The index of the element to retrieve.

        Returns:
            T: The element at the given index.

        Raises:
            IndexError: If index is out of bounds.
        """
        if not 0 <= index < self._size:
            raise IndexError("Index out of bounds.")
        return self._data[index]  # type: ignore

    def __setitem__(self, index: int, value: T) -> None:
        """
        Sets the element at the specified index to the given value.

        Args:
            index (int): The index of the element to set.
            value (T): The value to set at the specified index.

        Raises:
            IndexError: If index is out of bounds.
        """
        if not 0 <= index < self._size:
            raise IndexError("Index out of bounds.")
        self._data[index] = value

    def append(self, value: T) -> None:
        """
        Appends an element to the end of the array.

        Args:
            value (T): The value to append.
        """
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        self._data[self._size] = value
        self._size += 1

    def insert(self, index: int, value: T) -> None:
        """
        Inserts an element at the specified index.

        Args:
            index (int): The index at which to insert the element.
            value (T): The value to insert.

        Raises:
            IndexError: If index is out of bounds.
        """
        if not 0 <= index <= self._size:
            raise IndexError("Index out of bounds.")
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        for i in range(self._size, index, -1):
            self._data[i] = self._data[i - 1]
        self._data[index] = value
        self._size += 1

    def remove(self, value: T) -> None:
        """
        Removes the first occurrence of the specified value.

        Args:
            value (T): The value to remove.

        Raises:
            ValueError: If the value is not found.
        """
        index = self.index_of(value)
        if index == -1:
            raise ValueError("Value not found in array.")
        self.pop(index)

    def pop(self, index: int) -> T:
        """
        Removes and returns the element at the specified index.

        Args:
            index (int): The index of the element to remove.

        Returns:
            T: The removed element.

        Raises:
            IndexError: If index is out of bounds.
        """
        if not 0 <= index < self._size:
            raise IndexError("Index out of bounds.")
        value = self._data[index]
        for i in range(index, self._size - 1):
            self._data[i] = self._data[i + 1]
        self._data[self._size - 1] = None
        self._size -= 1
        if 0 < self._size < self._capacity // 4:
            self._resize(self._capacity // 2)
        return value  # type: ignore

    def index_of(self, value: T) -> int:
        """
        Returns the index of the first occurrence of the specified value.

        Args:
            value (T): The value to find.

        Returns:
            int: The index of the value, or -1 if not found.
        """
        for i in range(self._size):
            if self._data[i] == value:
                return i
        return -1

    def _resize(self, new_capacity: int) -> None:
        """
        Resizes the internal storage to a new capacity.

        Args:
            new_capacity (int): The new capacity.
        """
        new_data: List[Optional[T]] = [None] * new_capacity
        for i in range(self._size):
            new_data[i] = self._data[i]
        self._data = new_data
        self._capacity = new_capacity

    def clear(self) -> None:
        """Clears all elements from the array."""
        self._data = [None] * self._capacity
        self._size = 0

    def contains(self, value: T) -> bool:
        """
        Checks if the array contains the specified value.

        Args:
            value (T): The value to check.

        Returns:
            bool: True if the value is found, False otherwise.
        """
        return self.index_of(value) != -1

    def to_list(self) -> List[T]:
        """
        Converts the array to a standard Python list.

        Returns:
            List[T]: The list containing all elements of the array.
        """
        return [self._data[i] for i in range(self._size)]  # type: ignore


class ArrayAlgorithms:
    """
    A collection of static methods implementing common array algorithms.
    """

    @staticmethod
    def linear_search(arr: Array[T], target: T) -> int:
        """
        Performs a linear search for the target in the array.

        Args:
            arr (Array[T]): The array to search.
            target (T): The target value.

        Returns:
            int: The index of the target, or -1 if not found.
        """
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1

    @staticmethod
    def binary_search(arr: Array[T], target: T) -> int:
        """
        Performs a binary search for the target in a sorted array.

        Args:
            arr (Array[T]): The sorted array to search.
            target (T): The target value.

        Returns:
            int: The index of the target, or -1 if not found.
        """
        left: int = 0
        right: int = len(arr) - 1
        while left <= right:
            mid: int = left + (right - left) // 2
            mid_val = arr[mid]
            if mid_val == target:
                return mid
            elif mid_val < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    @staticmethod
    def bubble_sort(arr: Array[T]) -> None:
        """
        Sorts the array in ascending order using the bubble sort algorithm.

        Args:
            arr (Array[T]): The array to sort.
        """
        n = len(arr)
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
            if not swapped:
                break

    @staticmethod
    def selection_sort(arr: Array[T]) -> None:
        """
        Sorts the array in ascending order using the selection sort algorithm.

        Args:
            arr (Array[T]): The array to sort.
        """
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]

    @staticmethod
    def insertion_sort(arr: Array[T]) -> None:
        """
        Sorts the array in ascending order using the insertion sort algorithm.

        Args:
            arr (Array[T]): The array to sort.
        """
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    @staticmethod
    def quick_sort(arr: Array[T]) -> None:
        """
        Sorts the array in ascending order using the quick sort algorithm.

        Args:
            arr (Array[T]): The array to sort.
        """

        def _quick_sort(low: int, high: int) -> None:
            if low < high:
                pi = partition(low, high)
                _quick_sort(low, pi - 1)
                _quick_sort(pi + 1, high)

        def partition(low: int, high: int) -> int:
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            return i + 1

        _quick_sort(0, len(arr) - 1)

    @staticmethod
    def merge_sort(arr: Array[T]) -> None:
        """
        Sorts the array in ascending order using the merge sort algorithm.

        Args:
            arr (Array[T]): The array to sort.
        """

        def _merge_sort(left: int, right: int) -> None:
            if left < right:
                mid = (left + right) // 2
                _merge_sort(left, mid)
                _merge_sort(mid + 1, right)
                merge(left, mid, right)

        def merge(left: int, mid: int, right: int) -> None:
            n1 = mid - left + 1
            n2 = right - mid
            L: List[Optional[T]] = [arr[left + i] for i in range(n1)]  # type: ignore
            R: List[Optional[T]] = [arr[mid + 1 + j] for j in range(n2)]  # type: ignore
            i = j = 0
            k = left
            while i < n1 and j < n2:
                if L[i] is not None and R[j] is not None:
                    if L[i] <= R[j]:
                        arr[k] = L[i]  # type: ignore
                        i += 1
                    else:
                        arr[k] = R[j]  # type: ignore
                        j += 1
                k += 1
            while i < n1:
                arr[k] = L[i]  # type: ignore
                i += 1
                k += 1
            while j < n2:
                arr[k] = R[j]  # type: ignore
                j += 1
                k += 1

        _merge_sort(0, len(arr) - 1)


class MultiDimensionalArray:
    """
    A class representing a multi-dimensional array.
    """

    def __init__(self, dimensions: List[int]) -> None:
        """
        Initializes a multi-dimensional array with the specified dimensions.

        Args:
            dimensions (List[int]): A list representing the size of each dimension.

        Raises:
            ValueError: If any dimension size is non-positive.
        """
        if not dimensions:
            raise ValueError("Dimensions list cannot be empty.")
        for dim in dimensions:
            if dim <= 0:
                raise ValueError("All dimensions must be positive integers.")
        self.dimensions: List[int] = dimensions
        self.size: int = 1
        for dim in dimensions:
            self.size *= dim
        self.data: Array[Any] = Array[Any](self.size)

    def _get_flat_index(self, indices: List[int]) -> int:
        """
        Converts multi-dimensional indices to a flat index.

        Args:
            indices (List[int]): A list of indices for each dimension.

        Returns:
            int: The corresponding flat index.

        Raises:
            IndexError: If any index is out of bounds.
        """
        if len(indices) != len(self.dimensions):
            raise IndexError("Number of indices must match number of dimensions.")
        flat_index = 0
        multiplier = 1
        for i in reversed(range(len(self.dimensions))):
            if not 0 <= indices[i] < self.dimensions[i]:
                raise IndexError(f"Index {indices[i]} out of bounds for dimension {i}.")
            flat_index += indices[i] * multiplier
            multiplier *= self.dimensions[i]
        return flat_index

    def get(self, indices: List[int]) -> Any:
        """
        Retrieves the value at the specified multi-dimensional indices.

        Args:
            indices (List[int]): The multi-dimensional indices.

        Returns:
            Any: The value at the specified indices.
        """
        flat_index = self._get_flat_index(indices)
        return self.data[flat_index]

    def set(self, indices: List[int], value: Any) -> None:
        """
        Sets the value at the specified multi-dimensional indices.

        Args:
            indices (List[int]): The multi-dimensional indices.
            value (Any): The value to set.
        """
        flat_index = self._get_flat_index(indices)
        self.data[flat_index] = value

    def to_nested_list(self) -> Any:
        """
        Converts the multi-dimensional array to a nested Python list.

        Returns:
            Any: The nested list representation.
        """

        def build_nested_list(dim: int, start: int) -> Any:
            if dim == len(self.dimensions) - 1:
                return [self.data[i] for i in range(start, start + self.dimensions[dim])]
            nested = []
            step = 1
            for d in self.dimensions[dim + 1:]:
                step *= d
            for i in range(self.dimensions[dim]):
                nested.append(build_nested_list(dim + 1, start + i * step))
            return nested

        return build_nested_list(0, 0)


class SparseArray(Generic[T]):
    """
    A sparse array implementation using a dictionary to store non-zero elements.
    """

    def __init__(self, size: int) -> None:
        """
        Initializes a sparse array of a given size.

        Args:
            size (int): The size of the array.

        Raises:
            ValueError: If size is non-positive.
        """
        if size <= 0:
            raise ValueError("Size must be a positive integer.")
        self.size: int = size
        self.data: dict[int, T] = {}

    def __getitem__(self, index: int) -> T:
        """
        Retrieves the element at the specified index.

        Args:
            index (int): The index of the element.

        Returns:
            T: The element at the specified index.

        Raises:
            IndexError: If index is out of bounds.
        """
        if not 0 <= index < self.size:
            raise IndexError("Index out of bounds.")
        return self.data.get(index, 0)  # Assuming default value is 0

    def __setitem__(self, index: int, value: T) -> None:
        """
        Sets the element at the specified index to the given value.

        Args:
            index (int): The index of the element.
            value (T): The value to set.

        Raises:
            IndexError: If index is out of bounds.
        """
        if not 0 <= index < self.size:
            raise IndexError("Index out of bounds.")
        if value != 0:
            self.data[index] = value
        elif index in self.data:
            del self.data[index]

    def to_dense(self) -> List[T]:
        """
        Converts the sparse array to a dense list.

        Returns:
            List[T]: The dense list representation.
        """
        return [self.data.get(i, 0) for i in range(self.size)]  # Assuming default value is 0

    def non_zero_elements(self) -> List[tuple[int, T]]:
        """
        Returns a list of non-zero elements as tuples of (index, value).

        Returns:
            List[tuple[int, T]]: The list of non-zero elements.
        """
        return list(self.data.items())


class ArrayUtilities:
    """
    A utility class providing additional array-related functionalities.
    """

    @staticmethod
    def find_duplicates(arr: Array[T]) -> Array[T]:
        """
        Finds and returns all duplicate elements in the array.

        Args:
            arr (Array[T]): The array to check for duplicates.

        Returns:
            Array[T]: An array containing all duplicate elements.
        """
        seen: dict[T, int] = {}
        duplicates: Array[T] = Array()
        for i in range(len(arr)):
            elem = arr[i]
            if elem in seen:
                if seen[elem] == 1:
                    duplicates.append(elem)
                seen[elem] += 1
            else:
                seen[elem] = 1
        return duplicates

    @staticmethod
    def rotate_right(arr: Array[T], k: int) -> None:
        """
        Rotates the array to the right by k steps.

        Args:
            arr (Array[T]): The array to rotate.
            k (int): Number of steps to rotate.

        Raises:
            ValueError: If k is negative.
        """
        n = len(arr)
        if k < 0:
            raise ValueError("k must be non-negative.")
        k = k % n
        if k == 0:
            return
        ArrayUtilities._reverse(arr, 0, n - 1)
        ArrayUtilities._reverse(arr, 0, k - 1)
        ArrayUtilities._reverse(arr, k, n - 1)

    @staticmethod
    def _reverse(arr: Array[T], start: int, end: int) -> None:
        """
        Reverses elements in the array from start to end indices.

        Args:
            arr (Array[T]): The array to reverse.
            start (int): Starting index.
            end (int): Ending index.
        """
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1

    @staticmethod
    def maximum_subarray(arr: Array[T]) -> T:
        """
        Finds the contiguous subarray with the maximum sum using Kadane's algorithm.

        Args:
            arr (Array[T]): The array of integers.

        Returns:
            T: The maximum subarray sum.

        Raises:
            ValueError: If the array is empty.
        """
        if len(arr) == 0:
            raise ValueError("Array is empty.")
        max_so_far = arr[0]
        current_max = arr[0]
        for i in range(1, len(arr)):
            current_max = max(arr[i], current_max + arr[i])
            max_so_far = max(max_so_far, current_max)
        return max_so_far

    @staticmethod
    def merge_two_sorted_arrays(arr1: Array[T], arr2: Array[T]) -> Array[T]:
        """
        Merges two sorted arrays into a single sorted array.

        Args:
            arr1 (Array[T]): The first sorted array.
            arr2 (Array[T]): The second sorted array.

        Returns:
            Array[T]: The merged sorted array.
        """
        merged = Array[T](len(arr1) + len(arr2))
        i = j = 0
        while i < len(arr1) and j < len(arr2):
            if arr1[i] <= arr2[j]:
                merged.append(arr1[i])
                i += 1
            else:
                merged.append(arr2[j])
                j += 1
        while i < len(arr1):
            merged.append(arr1[i])
            i += 1
        while j < len(arr2):
            merged.append(arr2[j])
            j += 1
        return merged