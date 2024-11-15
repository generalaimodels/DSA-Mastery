"""
divide_and_conquer.py

A comprehensive exploration of the Divide and Conquer paradigm in algorithms,
ranging from basic concepts to advanced implementations. This module adheres to
PEP-8 standards, utilizes type hints for clarity, and emphasizes optimized,
robust, and maintainable Python code.

Author: OpenAI ChatGPT
Date: 2023-10-XX
"""

from typing import List, TypeVar, Callable
import torch

T = TypeVar('T')


class DivideAndConquer:
    """
    A class encapsulating various Divide and Conquer algorithms, demonstrating
    the implementation and application of this powerful algorithmic paradigm.
    """

    @staticmethod
    def merge_sort(arr: List[int]) -> List[int]:
        """
        Sorts an array of integers using the Merge Sort algorithm.

        Parameters:
            arr (List[int]): The list of integers to sort.

        Returns:
            List[int]: A new sorted list.
        """
        if len(arr) <= 1:
            return arr

        try:
            mid = len(arr) // 2
            left_half = DivideAndConquer.merge_sort(arr[:mid])
            right_half = DivideAndConquer.merge_sort(arr[mid:])
            return DivideAndConquer.merge(left_half, right_half)
        except Exception as e:
            raise RuntimeError(f"An error occurred during merge sort: {e}")

    @staticmethod
    def merge(left: List[int], right: List[int]) -> List[int]:
        """
        Merges two sorted lists into a single sorted list.

        Parameters:
            left (List[int]): The first sorted list.
            right (List[int]): The second sorted list.

        Returns:
            List[int]: Merged and sorted list.
        """
        merged = []
        i = j = 0

        try:
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    j += 1

            # Append any remaining elements
            merged.extend(left[i:])
            merged.extend(right[j:])
            return merged
        except Exception as e:
            raise RuntimeError(f"An error occurred during merging: {e}")

    @staticmethod
    def quick_sort(arr: List[int]) -> List[int]:
        """
        Sorts an array of integers using the Quick Sort algorithm.

        Parameters:
            arr (List[int]): The list of integers to sort.

        Returns:
            List[int]: A new sorted list.
        """
        if len(arr) <= 1:
            return arr

        try:
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            return DivideAndConquer.quick_sort(left) + middle + DivideAndConquer.quick_sort(right)
        except Exception as e:
            raise RuntimeError(f"An error occurred during quick sort: {e}")

    @staticmethod
    def binary_search(arr: List[int], target: int) -> int:
        """
        Searches for a target value within a sorted array using Binary Search.

        Parameters:
            arr (List[int]): The sorted list of integers.
            target (int): The integer value to search for.

        Returns:
            int: The index of the target if found; otherwise, -1.
        """
        try:
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
        except Exception as e:
            raise RuntimeError(f"An error occurred during binary search: {e}")

    @staticmethod
    def maximum_subarray(arr: List[int]) -> int:
        """
        Finds the maximum subarray sum using the Divide and Conquer approach.

        Parameters:
            arr (List[int]): The list of integers.

        Returns:
            int: The maximum subarray sum.
        """
        try:
            return DivideAndConquer._max_subarray_helper(arr, 0, len(arr) - 1)
        except Exception as e:
            raise RuntimeError(f"An error occurred during maximum subarray computation: {e}")

    @staticmethod
    def _max_subarray_helper(arr: List[int], left: int, right: int) -> int:
        """
        Helper function for maximum_subarray to recursively find the maximum sum.

        Parameters:
            arr (List[int]): The list of integers.
            left (int): The starting index.
            right (int): The ending index.

        Returns:
            int: The maximum subarray sum in the range [left, right].
        """
        if left == right:
            return arr[left]

        mid = (left + right) // 2
        max_left = DivideAndConquer._max_subarray_helper(arr, left, mid)
        max_right = DivideAndConquer._max_subarray_helper(arr, mid + 1, right)
        max_cross = DivideAndConquer._max_crossing_subarray(arr, left, mid, right)
        return max(max_left, max_right, max_cross)

    @staticmethod
    def _max_crossing_subarray(arr: List[int], left: int, mid: int, right: int) -> int:
        """
        Finds the maximum subarray sum that crosses the midpoint.

        Parameters:
            arr (List[int]): The list of integers.
            left (int): The starting index.
            mid (int): The midpoint index.
            right (int): The ending index.

        Returns:
            int: The maximum crossing subarray sum.
        """
        left_sum = float('-inf')
        total = 0
        for i in range(mid, left - 1, -1):
            total += arr[i]
            if total > left_sum:
                left_sum = total

        right_sum = float('-inf')
        total = 0
        for i in range(mid + 1, right + 1):
            total += arr[i]
            if total > right_sum:
                right_sum = total

        return left_sum + right_sum

    @staticmethod
    def matrix_multiplication(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
        """
        Multiplies two matrices using the Divide and Conquer approach.

        Parameters:
            a (List[List[int]]): The first matrix.
            b (List[List[int]]): The second matrix.

        Returns:
            List[List[int]]: The resulting matrix after multiplication.
        """
        try:
            n = len(a)
            # Base case when size of matrices is 1x1
            if n == 1:
                return [[a[0][0] * b[0][0]]]

            # Splitting the matrices into quadrants
            mid = n // 2
            a11, a12, a21, a22 = DivideAndConquer.split_matrix(a, mid)
            b11, b12, b21, b22 = DivideAndConquer.split_matrix(b, mid)

            # Recursively compute the products
            m1 = DivideAndConquer.matrix_multiplication(a11, b11)
            m2 = DivideAndConquer.matrix_multiplication(a12, b21)
            m3 = DivideAndConquer.matrix_multiplication(a11, b12)
            m4 = DivideAndConquer.matrix_multiplication(a12, b22)
            m5 = DivideAndConquer.matrix_multiplication(a21, b11)
            m6 = DivideAndConquer.matrix_multiplication(a22, b21)
            m7 = DivideAndConquer.matrix_multiplication(a21, b12)
            m8 = DivideAndConquer.matrix_multiplication(a22, b22)

            # Combine the results into a single matrix
            c11 = DivideAndConquer.add_matrix(m1, m2)
            c12 = DivideAndConquer.add_matrix(m3, m4)
            c21 = DivideAndConquer.add_matrix(m5, m6)
            c22 = DivideAndConquer.add_matrix(m7, m8)

            # Merge quadrants into a full matrix
            return DivideAndConquer.merge_quadrants(c11, c12, c21, c22)
        except Exception as e:
            raise RuntimeError(f"An error occurred during matrix multiplication: {e}")

    @staticmethod
    def split_matrix(matrix: List[List[int]], mid: int):
        """
        Splits a matrix into four quadrants.

        Parameters:
            matrix (List[List[int]]): The matrix to split.
            mid (int): The midpoint index.

        Returns:
            Tuple[List[List[int]], List[List[int]], List[List[int]], List[List[int]]]:
                The four quadrants of the matrix.
        """
        a11 = [row[:mid] for row in matrix[:mid]]
        a12 = [row[mid:] for row in matrix[:mid]]
        a21 = [row[:mid] for row in matrix[mid:]]
        a22 = [row[mid:] for row in matrix[mid:]]
        return a11, a12, a21, a22

    @staticmethod
    def add_matrix(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
        """
        Adds two matrices.

        Parameters:
            a (List[List[int]]): The first matrix.
            b (List[List[int]]): The second matrix.

        Returns:
            List[List[int]]: The resulting matrix after addition.
        """
        result = []
        try:
            for row_a, row_b in zip(a, b):
                result.append([x + y for x, y in zip(row_a, row_b)])
            return result
        except Exception as e:
            raise RuntimeError(f"An error occurred during matrix addition: {e}")

    @staticmethod
    def merge_quadrants(c11: List[List[int]], c12: List[List[int]],
                       c21: List[List[int]], c22: List[List[int]]) -> List[List[int]]:
        """
        Merges four quadrants into a single matrix.

        Parameters:
            c11, c12, c21, c22 (List[List[int]]): The four quadrants.

        Returns:
            List[List[int]]: The merged matrix.
        """
        merged = []
        try:
            for row1, row2 in zip(c11, c12):
                merged.append(row1 + row2)
            for row1, row2 in zip(c21, c22):
                merged.append(row1 + row2)
            return merged
        except Exception as e:
            raise RuntimeError(f"An error occurred during merging quadrants: {e}")

    @staticmethod
    def strassen_matrix_multiplication(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
        """
        Multiplies two matrices using Strassen's algorithm, an advanced
        Divide and Conquer technique.

        Parameters:
            a (List[List[int]]): The first matrix.
            b (List[List[int]]): The second matrix.

        Returns:
            List[List[int]]: The resulting matrix after multiplication.
        """
        try:
            n = len(a)
            # Base case for 1x1 matrix
            if n == 1:
                return [[a[0][0] * b[0][0]]]

            # Ensuring matrices are even-sized
            if n % 2 != 0:
                a = DivideAndConquer.pad_matrix(a)
                b = DivideAndConquer.pad_matrix(b)
                n += 1

            mid = n // 2
            a11, a12, a21, a22 = DivideAndConquer.split_matrix(a, mid)
            b11, b12, b21, b22 = DivideAndConquer.split_matrix(b, mid)

            # Strassen's 7 products
            m1 = DivideAndConquer.strassen_matrix_multiplication(
                DivideAndConquer.add_matrix(a11, a22),
                DivideAndConquer.add_matrix(b11, b22)
            )
            m2 = DivideAndConquer.strassen_matrix_multiplication(
                DivideAndConquer.add_matrix(a21, a22),
                b11
            )
            m3 = DivideAndConquer.strassen_matrix_multiplication(
                a11,
                DivideAndConquer.add_matrix(b12, b22)
            )
            m4 = DivideAndConquer.strassen_matrix_multiplication(
                a22,
                DivideAndConquer.add_matrix(b21, b11)
            )
            m5 = DivideAndConquer.strassen_matrix_multiplication(
                DivideAndConquer.add_matrix(a11, a12),
                b22
            )
            m6 = DivideAndConquer.strassen_matrix_multiplication(
                DivideAndConquer.subtract_matrix(a21, a11),
                DivideAndConquer.add_matrix(b11, b12)
            )
            m7 = DivideAndConquer.strassen_matrix_multiplication(
                DivideAndConquer.subtract_matrix(a12, a22),
                DivideAndConquer.add_matrix(b21, b22)
            )

            # Computing the final quadrants
            c11 = DivideAndConquer.add_matrix(
                DivideAndConquer.subtract_matrix(
                    DivideAndConquer.add_matrix(m1, m4), m5
                ),
                m7
            )
            c12 = DivideAndConquer.add_matrix(m3, m5)
            c21 = DivideAndConquer.add_matrix(m2, m4)
            c22 = DivideAndConquer.subtract_matrix(
                DivideAndConquer.add_matrix(m1, m3),
                DivideAndConquer.add_matrix(m2, m6)
            )

            # Merging the quadrants into a full matrix
            merged = DivideAndConquer.merge_quadrants(c11, c12, c21, c22)

            # Removing any padding if added
            if len(merged) > n or any(len(row) > n for row in merged):
                merged = [row[:n - 1] for row in merged[:n - 1]]

            return merged
        except Exception as e:
            raise RuntimeError(f"An error occurred during Strassen's multiplication: {e}")

    @staticmethod
    def subtract_matrix(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
        """
        Subtracts matrix b from matrix a.

        Parameters:
            a (List[List[int]]): The first matrix.
            b (List[List[int]]): The second matrix.

        Returns:
            List[List[int]]: The resulting matrix after subtraction.
        """
        result = []
        try:
            for row_a, row_b in zip(a, b):
                result.append([x - y for x, y in zip(row_a, row_b)])
            return result
        except Exception as e:
            raise RuntimeError(f"An error occurred during matrix subtraction: {e}")

    @staticmethod
    def pad_matrix(matrix: List[List[int]]) -> List[List[int]]:
        """
        Pads a matrix with zeros to make its dimensions even.

        Parameters:
            matrix (List[List[int]]): The matrix to pad.

        Returns:
            List[List[int]]: The padded matrix.
        """
        n = len(matrix)
        if n % 2 == 0:
            return matrix

        # Adding a new row of zeros
        new_row = [0] * n
        matrix.append(new_row)

        # Adding a new column of zeros to each row
        for row in matrix:
            row.append(0)

        return matrix

    @staticmethod
    def karatsuba_multiplication(x: int, y: int) -> int:
        """
        Multiplies two integers using the Karatsuba algorithm, an advanced
        Divide and Conquer technique.

        Parameters:
            x (int): The first integer.
            y (int): The second integer.

        Returns:
            int: The product of x and y.
        """
        try:
            # Base case for single-digit multiplication
            if x < 10 and y < 10:
                return x * y

            # Calculates the size of the numbers
            n = max(len(str(x)), len(str(y)))
            half = n // 2

            # Splitting the digit sequences
            high1, low1 = divmod(x, 10**half)
            high2, low2 = divmod(y, 10**half)

            # 3 recursive calls
            z0 = DivideAndConquer.karatsuba_multiplication(low1, low2)
            z1 = DivideAndConquer.karatsuba_multiplication((low1 + high1), (low2 + high2))
            z2 = DivideAndConquer.karatsuba_multiplication(high1, high2)

            return (z2 * 10**(2 * half)) + ((z1 - z2 - z0) * 10**half) + z0
        except Exception as e:
            raise RuntimeError(f"An error occurred during Karatsuba multiplication: {e}")


def main():
    """
    Demonstrates the usage of Divide and Conquer algorithms.
    """
    try:
        # Example for Merge Sort
        array = [38, 27, 43, 3, 9, 82, 10]
        sorted_array = DivideAndConquer.merge_sort(array)
        print(f"Merge Sort:\nOriginal: {array}\nSorted: {sorted_array}\n")

        # Example for Quick Sort
        array = [10, 7, 8, 9, 1, 5]
        sorted_array = DivideAndConquer.quick_sort(array)
        print(f"Quick Sort:\nOriginal: {array}\nSorted: {sorted_array}\n")

        # Example for Binary Search
        sorted_array = [1, 3, 5, 7, 9, 10, 27, 38, 43, 82]
        target = 27
        index = DivideAndConquer.binary_search(sorted_array, target)
        print(f"Binary Search:\nArray: {sorted_array}\nTarget: {target}\nIndex: {index}\n")

        # Example for Maximum Subarray
        array = [-2, -3, 4, -1, -2, 1, 5, -3]
        max_sum = DivideAndConquer.maximum_subarray(array)
        print(f"Maximum Subarray:\nArray: {array}\nMaximum Sum: {max_sum}\n")

        # Example for Matrix Multiplication
        matrix_a = [
            [1, 2],
            [3, 4]
        ]
        matrix_b = [
            [5, 6],
            [7, 8]
        ]
        product = DivideAndConquer.matrix_multiplication(matrix_a, matrix_b)
        print(f"Matrix Multiplication (Divide and Conquer):\nMatrix A: {matrix_a}\nMatrix B: {matrix_b}\nProduct: {product}\n")

        # Example for Strassen's Matrix Multiplication
        matrix_a = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]
        matrix_b = [
            [16, 15, 14, 13],
            [12, 11, 10, 9],
            [8, 7, 6, 5],
            [4, 3, 2, 1]
        ]
        product_strassen = DivideAndConquer.strassen_matrix_multiplication(matrix_a, matrix_b)
        print(f"Matrix Multiplication (Strassen's Algorithm):\nMatrix A: {matrix_a}\nMatrix B: {matrix_b}\nProduct: {product_strassen}\n")

        # Example for Karatsuba Multiplication
        x, y = 1234, 5678
        product_karatsuba = DivideAndConquer.karatsuba_multiplication(x, y)
        print(f"Karatsuba Multiplication:\nX: {x}\nY: {y}\nProduct: {product_karatsuba}\n")

    except Exception as e:
        print(f"An error occurred in main execution: {e}")


if __name__ == "__main__":
    main()