"""
searching_sorting.py

A comprehensive module covering various searching and sorting algorithms
from basic to advanced levels. This module adheres to PEP-8 standards,
utilizes type hints for clarity, and includes robust error handling.
"""

from typing import List, Optional, TypeVar, Generic

T = TypeVar('T')


class SearchingAlgorithms:
    """Class containing various searching algorithms."""

    @staticmethod
    def linear_search(arr: List[T], target: T) -> Optional[int]:
        """
        Perform a linear search for the target in the list.

        Args:
            arr (List[T]): The list to search.
            target (T): The target value to find.

        Returns:
            Optional[int]: The index of the target if found; otherwise, None.
        """
        if not arr:
            print("The array is empty.")
            return None

        for index, item in enumerate(arr):
            if item == target:
                print(f"Linear Search: Found target {target} at index {index}.")
                return index
        print(f"Linear Search: Target {target} not found.")
        return None

    @staticmethod
    def binary_search(arr: List[T], target: T) -> Optional[int]:
        """
        Perform a binary search for the target in the sorted list.

        Args:
            arr (List[T]): The sorted list to search.
            target (T): The target value to find.

        Returns:
            Optional[int]: The index of the target if found; otherwise, None.
        """
        if not arr:
            print("The array is empty.")
            return None

        left, right = 0, len(arr) - 1
        print(f"Binary Search: Searching for {target} in {arr}.")

        while left <= right:
            mid = left + (right - left) // 2
            mid_val = arr[mid]
            print(f"Binary Search: Checking middle index {mid} with value {mid_val}.")

            if mid_val == target:
                print(f"Binary Search: Found target {target} at index {mid}.")
                return mid
            elif mid_val < target:
                print(f"Binary Search: Target {target} is greater than {mid_val}.")
                left = mid + 1
            else:
                print(f"Binary Search: Target {target} is less than {mid_val}.")
                right = mid - 1

        print(f"Binary Search: Target {target} not found.")
        return None

    @staticmethod
    def jump_search(arr: List[T], target: T) -> Optional[int]:
        """
        Perform a jump search for the target in the sorted list.

        Args:
            arr (List[T]): The sorted list to search.
            target (T): The target value to find.

        Returns:
            Optional[int]: The index of the target if found; otherwise, None.
        """
        import math

        n = len(arr)
        if n == 0:
            print("The array is empty.")
            return None

        step = int(math.sqrt(n))
        prev = 0
        print(f"Jump Search: Searching for {target} with step size {step}.")

        while prev < n and arr[min(step, n) - 1] < target:
            print(f"Jump Search: Jumping from index {prev} to {step}.")
            prev = step
            step += int(math.sqrt(n))
            if prev >= n:
                print(f"Jump Search: Target {target} not found.")
                return None

        for index in range(prev, min(step, n)):
            print(f"Jump Search: Checking index {index} with value {arr[index]}.")
            if arr[index] == target:
                print(f"Jump Search: Found target {target} at index {index}.")
                return index

        print(f"Jump Search: Target {target} not found.")
        return None


class SortingAlgorithms:
    """Class containing various sorting algorithms."""

    @staticmethod
    def bubble_sort(arr: List[T]) -> None:
        """
        Perform bubble sort on the list.

        Args:
            arr (List[T]): The list to sort.

        Returns:
            None: The list is sorted in place.
        """
        n = len(arr)
        if n == 0:
            print("The array is empty. No sorting needed.")
            return

        print(f"Bubble Sort: Starting sort on {arr}.")
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                print(f"Bubble Sort: Comparing {arr[j]} and {arr[j + 1]}.")
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
                    print(f"Bubble Sort: Swapped to {arr}.")
            if not swapped:
                print("Bubble Sort: No swaps occurred. List is sorted.")
                break
        print(f"Bubble Sort: Finished sorting. Result: {arr}.")

    @staticmethod
    def insertion_sort(arr: List[T]) -> None:
        """
        Perform insertion sort on the list.

        Args:
            arr (List[T]): The list to sort.

        Returns:
            None: The list is sorted in place.
        """
        n = len(arr)
        if n == 0:
            print("The array is empty. No sorting needed.")
            return

        print(f"Insertion Sort: Starting sort on {arr}.")
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            print(f"Insertion Sort: Inserting {key} into the sorted part {arr[:i]}.")
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
                print(f"Insertion Sort: Shifted to {arr}.")
            arr[j + 1] = key
            print(f"Insertion Sort: Inserted {key}. Current list: {arr}.")
        print(f"Insertion Sort: Finished sorting. Result: {arr}.")

    @staticmethod
    def merge_sort(arr: List[T]) -> List[T]:
        """
        Perform merge sort on the list.

        Args:
            arr (List[T]): The list to sort.

        Returns:
            List[T]: A new sorted list.
        """
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        print(f"Merge Sort: Splitting {arr} into {arr[:mid]} and {arr[mid:]}.")
        left = SortingAlgorithms.merge_sort(arr[:mid])
        right = SortingAlgorithms.merge_sort(arr[mid:])

        merged = SortingAlgorithms._merge(left, right)
        print(f"Merge Sort: Merged {left} and {right} into {merged}.")
        return merged

    @staticmethod
    def _merge(left: List[T], right: List[T]) -> List[T]:
        """Helper method to merge two sorted lists."""
        merged = []
        i = j = 0
        print(f"Merge: Merging {left} and {right}.")

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                print(f"Merge: Appended {left[i]} from left.")
                i += 1
            else:
                merged.append(right[j])
                print(f"Merge: Appended {right[j]} from right.")
                j += 1

        while i < len(left):
            merged.append(left[i])
            print(f"Merge: Appended remaining {left[i]} from left.")
            i += 1

        while j < len(right):
            merged.append(right[j])
            print(f"Merge: Appended remaining {right[j]} from right.")
            j += 1

        return merged

    @staticmethod
    def quick_sort(arr: List[T]) -> None:
        """
        Perform quick sort on the list.

        Args:
            arr (List[T]): The list to sort.

        Returns:
            None: The list is sorted in place.
        """
        print(f"Quick Sort: Starting sort on {arr}.")
        SortingAlgorithms._quick_sort_helper(arr, 0, len(arr) - 1)
        print(f"Quick Sort: Finished sorting. Result: {arr}.")

    @staticmethod
    def _quick_sort_helper(arr: List[T], low: int, high: int) -> None:
        """Helper method for quick sort."""
        if low < high:
            pi = SortingAlgorithms._partition(arr, low, high)
            print(f"Quick Sort: Partitioned at index {pi}.")
            SortingAlgorithms._quick_sort_helper(arr, low, pi - 1)
            SortingAlgorithms._quick_sort_helper(arr, pi + 1, high)

    @staticmethod
    def _partition(arr: List[T], low: int, high: int) -> int:
        """Partition the array and return the partition index."""
        pivot = arr[high]
        i = low - 1
        print(f"Partition: Using pivot {pivot}.")

        for j in range(low, high):
            print(f"Partition: Comparing {arr[j]} with pivot {pivot}.")
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                print(f"Partition: Swapped {arr[i]} and {arr[j]}. Array now: {arr}.")

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        print(f"Partition: Swapped pivot to index {i + 1}. Array now: {arr}.")
        return i + 1

    @staticmethod
    def heap_sort(arr: List[T]) -> None:
        """
        Perform heap sort on the list.

        Args:
            arr (List[T]): The list to sort.

        Returns:
            None: The list is sorted in place.
        """
        n = len(arr)
        if n == 0:
            print("The array is empty. No sorting needed.")
            return

        print(f"Heap Sort: Starting sort on {arr}.")

        # Build a maxheap
        for i in range(n // 2 - 1, -1, -1):
            SortingAlgorithms._heapify(arr, n, i)

        # One by one extract elements
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            print(f"Heap Sort: Swapped {arr[i]} with {arr[0]}. Array now: {arr}.")
            SortingAlgorithms._heapify(arr, i, 0)

        print(f"Heap Sort: Finished sorting. Result: {arr}.")

    @staticmethod
    def _heapify(arr: List[T], n: int, i: int) -> None:
        """Helper method to heapify a subtree rooted at index i."""
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        print(f"Heapify: Heapifying subtree rooted at index {i}.")

        if left < n and arr[left] > arr[largest]:
            largest = left
            print(f"Heapify: Left child {arr[left]} is larger than current largest {arr[largest]}.")

        if right < n and arr[right] > arr[largest]:
            largest = right
            print(f"Heapify: Right child {arr[right]} is larger than current largest {arr[largest]}.")

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            print(f"Heapify: Swapped {arr[i]} with {arr[largest]}. Array now: {arr}.")
            SortingAlgorithms._heapify(arr, n, largest)


# Example Usage
if __name__ == "__main__":
    # Sample data for demonstration
    data = [64, 34, 25, 12, 22, 11, 90]
    target = 22

    # Searching Algorithms
    print("=== Searching Algorithms ===")
    index = SearchingAlgorithms.linear_search(data, target)
    index = SearchingAlgorithms.binary_search(sorted(data), target)
    index = SearchingAlgorithms.jump_search(sorted(data), target)

    # Sorting Algorithms
    print("\n=== Sorting Algorithms ===")
    unsorted_data = data.copy()
    SortingAlgorithms.bubble_sort(unsorted_data)
    unsorted_data = data.copy()
    SortingAlgorithms.insertion_sort(unsorted_data)
    sorted_data = SortingAlgorithms.merge_sort(data.copy())
    print(f"Merge Sort Result: {sorted_data}")
    unsorted_data = data.copy()
    SortingAlgorithms.quick_sort(unsorted_data)
    unsorted_data = data.copy()
    SortingAlgorithms.heap_sort(unsorted_data)