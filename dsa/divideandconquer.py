"""
Divide and Conquer Algorithms Module

This module provides a comprehensive suite of divide and conquer algorithms,
ranging from basic to advanced levels. Each algorithm is implemented with
attention to efficiency, readability, and robustness, adhering to PEP-8 standards
and utilizing Python's typing module for clarity.

Algorithms Included:
1. Binary Search
2. Merge Sort
3. Quick Sort
4. Closest Pair of Points
5. Strassen's Matrix Multiplication

Author: OpenAI ChatGPT
Date: 2023-10
"""

from typing import List, Any, Optional, Tuple


class DivideAndConquer:
    """
    Base class for Divide and Conquer algorithms.
    """

    @staticmethod
    def binary_search(arr: List[int], target: int) -> int:
        """
        Perform binary search on a sorted array to find the target value.

        Args:
            arr (List[int]): Sorted list of integers.
            target (int): Value to search for.

        Returns:
            int: Index of target if found, else -1.

        Raises:
            ValueError: If the input array is empty.
        """
        if not arr:
            raise ValueError("The input array is empty.")

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
        return -1

    @staticmethod
    def merge_sort(arr: List[int]) -> List[int]:
        """
        Sort the array using Merge Sort algorithm.

        Args:
            arr (List[int]): List of integers to sort.

        Returns:
            List[int]: Sorted list of integers.

        Raises:
            ValueError: If the input array is None.
        """
        if arr is None:
            raise ValueError("Input array cannot be None.")

        def merge(left: List[int], right: List[int]) -> List[int]:
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

        def divide_and_conquer(arr_subset: List[int]) -> List[int]:
            if len(arr_subset) <= 1:
                return arr_subset
            mid = len(arr_subset) // 2
            left = divide_and_conquer(arr_subset[:mid])
            right = divide_and_conquer(arr_subset[mid:])
            return merge(left, right)

        return divide_and_conquer(arr)

    @staticmethod
    def quick_sort(arr: List[int]) -> List[int]:
        """
        Sort the array using Quick Sort algorithm.

        Args:
            arr (List[int]): List of integers to sort.

        Returns:
            List[int]: Sorted list of integers.

        Raises:
            ValueError: If the input array is None.
        """
        if arr is None:
            raise ValueError("Input array cannot be None.")

        def _quick_sort(items: List[int], low: int, high: int) -> None:
            if low < high:
                pivot_index = partition(items, low, high)
                _quick_sort(items, low, pivot_index - 1)
                _quick_sort(items, pivot_index + 1, high)

        def partition(items: List[int], low: int, high: int) -> int:
            pivot = items[high]
            i = low - 1
            for j in range(low, high):
                if items[j] <= pivot:
                    i += 1
                    items[i], items[j] = items[j], items[i]
            items[i + 1], items[high] = items[high], items[i + 1]
            return i + 1

        items_copy = arr.copy()
        _quick_sort(items_copy, 0, len(items_copy) - 1)
        return items_copy

    @staticmethod
    def closest_pair_of_points(points: List[Tuple[int, int]]) -> float:
        """
        Find the smallest distance between two points in a set of points.

        Args:
            points (List[Tuple[int, int]]): List of (x, y) coordinates.

        Returns:
            float: Minimum distance found.

        Raises:
            ValueError: If there are fewer than two points.
        """
        if points is None or len(points) < 2:
            raise ValueError("At least two points are required to compute distance.")

        def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        def closest_pair_recursive(px: List[Tuple[int, int]], py: List[Tuple[int, int]]) -> float:
            n = len(px)
            if n <= 3:
                min_dist = float('inf')
                for i in range(n):
                    for j in range(i + 1, n):
                        min_dist = min(min_dist, distance(px[i], px[j]))
                return min_dist

            mid = n // 2
            Qx = px[:mid]
            Rx = px[mid:]

            midpoint = px[mid][0]
            Qy = list(filter(lambda p: p[0] <= midpoint, py))
            Ry = list(filter(lambda p: p[0] > midpoint, py))

            dist_left = closest_pair_recursive(Qx, Qy)
            dist_right = closest_pair_recursive(Rx, Ry)
            delta = min(dist_left, dist_right)

            strip = [p for p in py if abs(p[0] - midpoint) < delta]
            min_dist_strip = delta
            for i in range(len(strip)):
                for j in range(i + 1, min(i + 7, len(strip))):
                    min_dist_strip = min(min_dist_strip, distance(strip[i], strip[j]))
            return min(min_dist_strip, delta)

        px = sorted(points, key=lambda p: p[0])
        py = sorted(points, key=lambda p: p[1])
        return closest_pair_recursive(px, py)

    @staticmethod
    def strassen_matrix_multiplication(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        """
        Multiply two matrices using Strassen's algorithm.

        Args:
            A (List[List[int]]): First matrix.
            B (List[List[int]]): Second matrix.

        Returns:
            List[List[int]]: Resultant matrix after multiplication.

        Raises:
            ValueError: If matrices are not square or have mismatched dimensions.
        """
        def next_power_of_two(n: int) -> int:
            return 1 if n == 0 else 2**(n - 1).bit_length()

        def add_matrix(X: List[List[int]], Y: List[List[int]]) -> List[List[int]]:
            n = len(X)
            return [[X[i][j] + Y[i][j] for j in range(n)] for i in range(n)]

        def subtract_matrix(X: List[List[int]], Y: List[List[int]]) -> List[List[int]]:
            n = len(X)
            return [[X[i][j] - Y[i][j] for j in range(n)] for i in range(n)]

        def split_matrix(M: List[List[int]]) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[List[int]]]:
            n = len(M)
            mid = n // 2
            A = [row[:mid] for row in M[:mid]]
            B = [row[mid:] for row in M[:mid]]
            C = [row[:mid] for row in M[mid:]]
            D = [row[mid:] for row in M[mid:]]
            return A, B, C, D

        def strassen(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
            n = len(A)
            if n == 1:
                return [[A[0][0] * B[0][0]]]
            A11, A12, A21, A22 = split_matrix(A)
            B11, B12, B21, B22 = split_matrix(B)

            M1 = strassen(add_matrix(A11, A22), add_matrix(B11, B22))
            M2 = strassen(add_matrix(A21, A22), B11)
            M3 = strassen(A11, subtract_matrix(B12, B22))
            M4 = strassen(A22, subtract_matrix(B21, B11))
            M5 = strassen(add_matrix(A11, A12), B22)
            M6 = strassen(subtract_matrix(A21, A11), add_matrix(B11, B12))
            M7 = strassen(subtract_matrix(A12, A22), add_matrix(B21, B22))

            C11 = add_matrix(subtract_matrix(add_matrix(M1, M4), M5), M7)
            C12 = add_matrix(M3, M5)
            C21 = add_matrix(M2, M4)
            C22 = add_matrix(subtract_matrix(add_matrix(M1, M3), M2), M6)

            new_matrix = []
            for i in range(len(C11)):
                new_matrix.append(C11[i] + C12[i])
            for i in range(len(C21)):
                new_matrix.append(C21[i] + C22[i])
            return new_matrix

        if not A or not B:
            raise ValueError("Input matrices cannot be empty.")

        n = len(A)
        if any(len(row) != n for row in A) or any(len(row) != n for row in B):
            raise ValueError("Matrices must be square and of the same dimensions.")

        m = next_power_of_two(n)
        A_padded = [row + [0] * (m - n) for row in A] + [[0] * m for _ in range(m - n)]
        B_padded = [row + [0] * (m - n) for row in B] + [[0] * m for _ in range(m - n)]

        C_padded = strassen(A_padded, B_padded)
        C = [row[:n] for row in C_padded[:n]]
        return C


def main():
    """
    Main function to demonstrate Divide and Conquer algorithms.
    """
    dac = DivideAndConquer()

    # Binary Search Example
    try:
        sorted_array = [1, 3, 5, 7, 9, 11]
        target = 7
        index = dac.binary_search(sorted_array, target)
        print(f"Binary Search: Element {target} found at index {index}.")
    except ValueError as ve:
        print(f"Binary Search Error: {ve}")

    # Merge Sort Example
    try:
        unsorted_array = [38, 27, 43, 3, 9, 82, 10]
        sorted_arr = dac.merge_sort(unsorted_array)
        print(f"Merge Sort: {sorted_arr}")
    except ValueError as ve:
        print(f"Merge Sort Error: {ve}")

    # Quick Sort Example
    try:
        unsorted_array_qs = [10, 7, 8, 9, 1, 5]
        sorted_qs = dac.quick_sort(unsorted_array_qs)
        print(f"Quick Sort: {sorted_qs}")
    except ValueError as ve:
        print(f"Quick Sort Error: {ve}")

    # Closest Pair of Points Example
    try:
        points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
        min_dist = dac.closest_pair_of_points(points)
        print(f"Closest Pair of Points: Minimum distance is {min_dist}.")
    except ValueError as ve:
        print(f"Closest Pair of Points Error: {ve}")

    # Strassen's Matrix Multiplication Example
    try:
        A = [
            [1, 2],
            [3, 4]
        ]
        B = [
            [5, 6],
            [7, 8]
        ]
        C = dac.strassen_matrix_multiplication(A, B)
        print("Strassen's Matrix Multiplication Result:")
        for row in C:
            print(row)
    except ValueError as ve:
        print(f"Strassen's Matrix Multiplication Error: {ve}")


if __name__ == "__main__":
    main()