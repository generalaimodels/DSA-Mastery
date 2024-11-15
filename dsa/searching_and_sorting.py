"""
searching_and_sorting.py

A comprehensive module demonstrating various searching and sorting algorithms
ranging from basic to advanced levels. Adheres to PEP-8 standards, utilizes type hints,
and includes robust error handling. Designed for clarity, maintainability, and scalability.


Author: Generalmodelai-agent
Date: 2024-10-15
"""

from typing import List, TypeVar, Generic, Optional

T = TypeVar('T')


class SearchAlgorithms:
    """Class encapsulating various searching algorithms."""

    @staticmethod
    def linear_search(arr: List[T], target: T) -> Optional[int]:
        """
        Perform linear search on a list.

        Args:
            arr (List[T]): The list to search.
            target (T): The target value to find.

        Returns:
            Optional[int]: The index of the target if found; otherwise, None.
        """
        if not isinstance(arr, list):
            raise TypeError("The first argument must be a list.")
        for index, value in enumerate(arr):
            if value == target:
                return index
        return None

    @staticmethod
    def binary_search(arr: List[T], target: T) -> Optional[int]:
        """
        Perform binary search on a sorted list.

        Args:
            arr (List[T]): The sorted list to search.
            target (T): The target value to find.

        Returns:
            Optional[int]: The index of the target if found; otherwise, None.
        """
        if not isinstance(arr, list):
            raise TypeError("The first argument must be a list.")
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            mid_val = arr[mid]
            if mid_val == target:
                return mid
            elif mid_val < target:
                left = mid + 1
            else:
                right = mid - 1
        return None

    @staticmethod
    def jump_search(arr: List[T], target: T) -> Optional[int]:
        """
        Perform jump search on a sorted list.

        Args:
            arr (List[T]): The sorted list to search.
            target (T): The target value to find.

        Returns:
            Optional[int]: The index of the target if found; otherwise, None.
        """
        import math

        if not isinstance(arr, list):
            raise TypeError("The first argument must be a list.")
        n = len(arr)
        step = int(math.sqrt(n))
        prev = 0

        while prev < n and arr[min(step, n) - 1] < target:
            prev = step
            step += int(math.sqrt(n))
            if prev >= n:
                return None

        for index in range(prev, min(step, n)):
            if arr[index] == target:
                return index
        return None

    @staticmethod
    def interpolation_search(arr: List[T], target: T) -> Optional[int]:
        """
        Perform interpolation search on a sorted list with uniformly distributed values.

        Args:
            arr (List[T]): The sorted list to search.
            target (T): The target value to find.

        Returns:
            Optional[int]: The index of the target if found; otherwise, None.
        """
        if not isinstance(arr, list):
            raise TypeError("The first argument must be a list.")
        low, high = 0, len(arr) - 1

        while low <= high and arr[low] <= target <= arr[high]:
            if low == high:
                if arr[low] == target:
                    return low
                return None
            # Estimate the position
            pos = low + int(((float(high - low) / (arr[high] - arr[low])) * (target - arr[low])))
            if pos < 0 or pos >= len(arr):
                return None
            if arr[pos] == target:
                return pos
            if arr[pos] < target:
                low = pos + 1
            else:
                high = pos - 1
        return None

    @staticmethod
    def exponential_search(arr: List[T], target: T) -> Optional[int]:
        """
        Perform exponential search on a sorted list.

        Args:
            arr (List[T]): The sorted list to search.
            target (T): The target value to find.

        Returns:
            Optional[int]: The index of the target if found; otherwise, None.
        """
        if not isinstance(arr, list):
            raise TypeError("The first argument must be a list.")

        if not arr:
            return None

        if arr[0] == target:
            return 0

        n = len(arr)
        i = 1
        while i < n and arr[i] <= target:
            i *= 2

        # Binary search between i/2 and min(i, n)
        left = i // 2
        right = min(i, n)

        return SearchAlgorithms.binary_search(arr[left:right], target) if arr[left:right] else None


class SortAlgorithms:
    """Class encapsulating various sorting algorithms."""

    @staticmethod
    def bubble_sort(arr: List[T]) -> List[T]:
        """
        Perform bubble sort on a list.

        Args:
            arr (List[T]): The list to sort.

        Returns:
            List[T]: The sorted list.
        """
        if not isinstance(arr, list):
            raise TypeError("The argument must be a list.")
        n = len(arr)
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
            if not swapped:
                break
        return arr

    @staticmethod
    def selection_sort(arr: List[T]) -> List[T]:
        """
        Perform selection sort on a list.

        Args:
            arr (List[T]): The list to sort.

        Returns:
            List[T]: The sorted list.
        """
        if not isinstance(arr, list):
            raise TypeError("The argument must be a list.")
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr

    @staticmethod
    def insertion_sort(arr: List[T]) -> List[T]:
        """
        Perform insertion sort on a list.

        Args:
            arr (List[T]): The list to sort.

        Returns:
            List[T]: The sorted list.
        """
        if not isinstance(arr, list):
            raise TypeError("The argument must be a list.")
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr

    @staticmethod
    def merge_sort(arr: List[T]) -> List[T]:
        """
        Perform merge sort on a list.

        Args:
            arr (List[T]): The list to sort.

        Returns:
            List[T]: The sorted list.
        """
        if not isinstance(arr, list):
            raise TypeError("The argument must be a list.")

        def _merge_sort(sub_arr: List[T]) -> List[T]:
            if len(sub_arr) <= 1:
                return sub_arr
            mid = len(sub_arr) // 2
            left = _merge_sort(sub_arr[:mid])
            right = _merge_sort(sub_arr[mid:])
            return SortAlgorithms._merge(left, right)

        return _merge_sort(arr)

    @staticmethod
    def _merge(left: List[T], right: List[T]) -> List[T]:
        """Merge two sorted lists into one sorted list."""
        merged = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged

    @staticmethod
    def quick_sort(arr: List[T]) -> List[T]:
        """
        Perform quick sort on a list.

        Args:
            arr (List[T]): The list to sort.

        Returns:
            List[T]: The sorted list.
        """
        if not isinstance(arr, list):
            raise TypeError("The argument must be a list.")

        def _quick_sort(items: List[T], low: int, high: int) -> None:
            if low < high:
                pi = SortAlgorithms._partition(items, low, high)
                _quick_sort(items, low, pi - 1)
                _quick_sort(items, pi + 1, high)

        _quick_sort(arr, 0, len(arr) - 1)
        return arr

    @staticmethod
    def _partition(items: List[T], low: int, high: int) -> int:
        """Partition the list and return the pivot index."""
        pivot = items[high]
        i = low - 1
        for j in range(low, high):
            if items[j] <= pivot:
                i += 1
                items[i], items[j] = items[j], items[i]
        items[i + 1], items[high] = items[high], items[i + 1]
        return i + 1

    @staticmethod
    def heap_sort(arr: List[T]) -> List[T]:
        """
        Perform heap sort on a list.

        Args:
            arr (List[T]): The list to sort.

        Returns:
            List[T]: The sorted list.
        """
        if not isinstance(arr, list):
            raise TypeError("The argument must be a list.")

        n = len(arr)

        def _heapify(items: List[T], n: int, i: int) -> None:
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and items[left] > items[largest]:
                largest = left
            if right < n and items[right] > items[largest]:
                largest = right
            if largest != i:
                items[i], items[largest] = items[largest], items[i]
                _heapify(items, n, largest)

        # Build a maxheap
        for i in range(n // 2 - 1, -1, -1):
            _heapify(arr, n, i)

        # Extract elements
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            _heapify(arr, i, 0)

        return arr

    @staticmethod
    def counting_sort(arr: List[int]) -> List[int]:
        """
        Perform counting sort on a list of integers.

        Args:
            arr (List[int]): The list to sort.

        Returns:
            List[int]: The sorted list.
        """
        if not isinstance(arr, list):
            raise TypeError("The argument must be a list.")
        if not all(isinstance(x, int) for x in arr):
            raise ValueError("All elements must be integers for counting sort.")

        if not arr:
            return arr

        max_val = max(arr)
        min_val = min(arr)
        range_of_elements = max_val - min_val + 1
        count = [0] * range_of_elements
        output = [0] * len(arr)

        for number in arr:
            count[number - min_val] += 1
        for i in range(1, len(count)):
            count[i] += count[i - 1]
        for number in reversed(arr):
            output[count[number - min_val] - 1] = number
            count[number - min_val] -= 1
        return output

    @staticmethod
    def radix_sort(arr: List[int]) -> List[int]:
        """
        Perform radix sort on a list of non-negative integers.

        Args:
            arr (List[int]): The list to sort.

        Returns:
            List[int]: The sorted list.
        """
        if not isinstance(arr, list):
            raise TypeError("The argument must be a list.")
        if not all(isinstance(x, int) and x >= 0 for x in arr):
            raise ValueError("All elements must be non-negative integers for radix sort.")

        if not arr:
            return arr

        max_val = max(arr)
        exp = 1
        while max_val // exp > 0:
            arr = SortAlgorithms._counting_sort_by_digit(arr, exp)
            exp *= 10
        return arr

    @staticmethod
    def _counting_sort_by_digit(arr: List[int], exp: int) -> List[int]:
        """Helper function for radix sort to perform counting sort based on digit."""
        n = len(arr)
        output = [0] * n
        count = [0] * 10

        for number in arr:
            index = (number // exp) % 10
            count[index] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        for number in reversed(arr):
            index = (number // exp) % 10
            output[count[index] - 1] = number
            count[index] -= 1
        return output

    @staticmethod
    def shell_sort(arr: List[T]) -> List[T]:
        """
        Perform shell sort on a list.

        Args:
            arr (List[T]): The list to sort.

        Returns:
            List[T]: The sorted list.
        """
        if not isinstance(arr, list):
            raise TypeError("The argument must be a list.")
        n = len(arr)
        gap = n // 2

        while gap > 0:
            for i in range(gap, n):
                temp = arr[i]
                j = i
                while j >= gap and arr[j - gap] > temp:
                    arr[j] = arr[j - gap]
                    j -= gap
                arr[j] = temp
            gap //= 2
        return arr

    @staticmethod
    def tim_sort(arr: List[T]) -> List[T]:
        """
        Perform Timsort on a list. Python's built-in sort uses Timsort.

        Args:
            arr (List[T]): The list to sort.

        Returns:
            List[T]: The sorted list.
        """
        if not isinstance(arr, list):
            raise TypeError("The argument must be a list.")
        return sorted(arr)

    @staticmethod
    def bucket_sort(arr: List[float], num_buckets: int = 10) -> List[float]:
        """
        Perform bucket sort on a list of floats.

        Args:
            arr (List[float]): The list to sort.
            num_buckets (int): Number of buckets to use.

        Returns:
            List[float]: The sorted list.
        """
        if not isinstance(arr, list):
            raise TypeError("The first argument must be a list.")
        if not all(isinstance(x, float) for x in arr):
            raise ValueError("All elements must be floats for bucket sort.")
        if num_buckets <= 0:
            raise ValueError("Number of buckets must be a positive integer.")

        if not arr:
            return arr

        min_val, max_val = min(arr), max(arr)
        bucket_range = (max_val - min_val) / num_buckets
        buckets = [[] for _ in range(num_buckets)]

        for number in arr:
            index = min(int((number - min_val) / bucket_range), num_buckets - 1)
            buckets[index].append(number)

        sorted_arr = []
        for bucket in buckets:
            sorted_arr.extend(SortAlgorithms.insertion_sort(bucket))
        return sorted_arr


# Example Usage and Test Cases
if __name__ == "__main__":
    # Test Searching Algorithms
    search_list = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7
    print("Linear Search:", SearchAlgorithms.linear_search(search_list, target))
    print("Binary Search:", SearchAlgorithms.binary_search(search_list, target))
    print("Jump Search:", SearchAlgorithms.jump_search(search_list, target))
    print("Interpolation Search:", SearchAlgorithms.interpolation_search(search_list, target))
    print("Exponential Search:", SearchAlgorithms.exponential_search(search_list, target))

    # Test Sorting Algorithms
    unsorted_list = [64, 34, 25, 12, 22, 11, 90]
    print("Bubble Sort:", SortAlgorithms.bubble_sort(unsorted_list.copy()))
    print("Selection Sort:", SortAlgorithms.selection_sort(unsorted_list.copy()))
    print("Insertion Sort:", SortAlgorithms.insertion_sort(unsorted_list.copy()))
    print("Merge Sort:", SortAlgorithms.merge_sort(unsorted_list.copy()))
    print("Quick Sort:", SortAlgorithms.quick_sort(unsorted_list.copy()))
    print("Heap Sort:", SortAlgorithms.heap_sort(unsorted_list.copy()))
    print("Counting Sort:", SortAlgorithms.counting_sort([4, 2, 2, 8, 3, 3, 1]))
    print("Radix Sort:", SortAlgorithms.radix_sort([170, 45, 75, 90, 802, 24, 2, 66]))
    print("Shell Sort:", SortAlgorithms.shell_sort(unsorted_list.copy()))
    print("Tim Sort:", SortAlgorithms.tim_sort(unsorted_list.copy()))
    print("Bucket Sort:", SortAlgorithms.bucket_sort([0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434]))