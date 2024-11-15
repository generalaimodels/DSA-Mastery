"""
Recursion in Data Structures and Algorithms
==========================================

This module provides comprehensive examples of recursion in various contexts,
ranging from basic to advanced levels. Each example includes type hints, adheres
to PEP-8 standards, and incorporates robust error handling for maintainability
and scalability.

Modules Used:
-------------
- torch: Used exclusively for tensor operations where applicable.

Author: Generalmodelai-agent
Date: 2024-10-15
"""

from typing import List, Any, Optional, Callable
import torch


def factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer n using recursion.

    Parameters:
    -----------
    n : int
        A non-negative integer whose factorial is to be computed.

    Returns:
    --------
    int
        Factorial of the input integer n.

    Raises:
    -------
    ValueError
        If n is negative.
    TypeError
        If n is not an integer.
    """
    if not isinstance(n, int):
        raise TypeError(f"Expected integer, got {type(n).__name__} instead.")
    if n < 0:
        raise ValueError("Factorial is not defined for negative integers.")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)


def fibonacci(n: int) -> int:
    """
    Compute the nth Fibonacci number using recursion with memoization.

    Parameters:
    -----------
    n : int
        The position in the Fibonacci sequence.

    Returns:
    --------
    int
        The nth Fibonacci number.

    Raises:
    -------
    ValueError
        If n is negative.
    TypeError
        If n is not an integer.
    """
    if not isinstance(n, int):
        raise TypeError(f"Expected integer, got {type(n).__name__} instead.")
    if n < 0:
        raise ValueError("Fibonacci number is not defined for negative integers.")
    
    memo: dict = {}

    def _fibonacci(m: int) -> int:
        if m in memo:
            return memo[m]
        if m == 0:
            memo[m] = 0
        elif m == 1:
            memo[m] = 1
        else:
            memo[m] = _fibonacci(m - 1) + _fibonacci(m - 2)
        return memo[m]

    return _fibonacci(n)


def quicksort(arr: List[int]) -> List[int]:
    """
    Sort an array of integers using the quicksort algorithm.

    Parameters:
    -----------
    arr : List[int]
        The list of integers to be sorted.

    Returns:
    --------
    List[int]
        A new sorted list.

    Raises:
    -------
    TypeError
        If arr is not a list of integers.
    """
    if not isinstance(arr, list):
        raise TypeError(f"Expected list, got {type(arr).__name__} instead.")
    if not all(isinstance(x, int) for x in arr):
        raise TypeError("All elements in the list must be integers.")
    if len(arr) <= 1:
        return arr.copy()
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


def binary_search(arr: List[int], target: int, low: int = 0, high: Optional[int] = None) -> int:
    """
    Perform binary search on a sorted array to find the index of the target element.

    Parameters:
    -----------
    arr : List[int]
        The sorted list of integers.
    target : int
        The integer to search for.
    low : int, optional
        The lower index of the search interval, by default 0.
    high : Optional[int], optional
        The higher index of the search interval, by default None.

    Returns:
    --------
    int
        The index of the target element if found; otherwise, -1.

    Raises:
    -------
    TypeError
        If arr is not a list of integers or target is not an integer.
    ValueError
        If the list is empty.
    """
    if not isinstance(arr, list):
        raise TypeError(f"Expected list, got {type(arr).__name__} instead.")
    if not all(isinstance(x, int) for x in arr):
        raise TypeError("All elements in the list must be integers.")
    if not isinstance(target, int):
        raise TypeError(f"Expected integer for target, got {type(target).__name__} instead.")
    if high is None:
        high = len(arr) - 1
    if low > high:
        return -1
    mid = (low + high) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search(arr, target, mid + 1, high)
    else:
        return binary_search(arr, target, low, mid - 1)


def generate_permutations(s: str) -> List[str]:
    """
    Generate all permutations of a given string using recursion.

    Parameters:
    -----------
    s : str
        The input string.

    Returns:
    --------
    List[str]
        A list containing all permutations of the input string.

    Raises:
    -------
    TypeError
        If s is not a string.
    """
    if not isinstance(s, str):
        raise TypeError(f"Expected string, got {type(s).__name__} instead.")
    if len(s) == 0:
        return [""]

    def _permute(chars: List[str], current: str, result: List[str]) -> None:
        if not chars:
            result.append(current)
            return
        for i in range(len(chars)):
            _permute(chars[:i] + chars[i+1:], current + chars[i], result)

    result: List[str] = []
    _permute(list(s), "", result)
    return result


class BinaryTreeNode:
    """
    A node in a binary tree.

    Attributes:
    -----------
    value : Any
        The value stored in the node.
    left : Optional['BinaryTreeNode']
        Reference to the left child node.
    right : Optional['BinaryTreeNode']
        Reference to the right child node.
    """

    def __init__(self, value: Any) -> None:
        self.value = value
        self.left: Optional[BinaryTreeNode] = None
        self.right: Optional[BinaryTreeNode] = None


def inorder_traversal(root: Optional[BinaryTreeNode]) -> List[Any]:
    """
    Perform in-order traversal of a binary tree.

    Parameters:
    -----------
    root : Optional[BinaryTreeNode]
        The root node of the binary tree.

    Returns:
    --------
    List[Any]
        A list of values representing the in-order traversal.

    Raises:
    -------
    TypeError
        If root is not a BinaryTreeNode or None.
    """

    if root is not None and not isinstance(root, BinaryTreeNode):
        raise TypeError("root must be a BinaryTreeNode or None.")

    def _inorder(node: Optional[BinaryTreeNode], acc: List[Any]) -> None:
        if node is not None:
            _inorder(node.left, acc)
            acc.append(node.value)
            _inorder(node.right, acc)

    result: List[Any] = []
    _inorder(root, result)
    return result


def solve_n_queens(n: int) -> List[List[str]]:
    """
    Solve the N-Queens problem and return all distinct solutions.

    Each solution contains a distinct board configuration of the n-queens' placement.

    Parameters:
    -----------
    n : int
        The number of queens and the size of the board (n x n).

    Returns:
    --------
    List[List[str]]
        A list of solutions, where each solution is a list of strings representing the board.

    Raises:
    -------
    ValueError
        If n is less than 1.
    TypeError
        If n is not an integer.
    """
    if not isinstance(n, int):
        raise TypeError(f"Expected integer, got {type(n).__name__} instead.")
    if n < 1:
        raise ValueError("Number of queens must be at least 1.")

    solutions: List[List[str]] = []
    board: List[int] = []
    cols: set = set()
    diags: set = set()
    anti_diags: set = set()

    def _backtrack(row: int) -> None:
        if row == n:
            solution = []
            for i in board:
                line = '.' * i + 'Q' + '.' * (n - i - 1)
                solution.append(line)
            solutions.append(solution)
            return
        for col in range(n):
            if col in cols or (row - col) in diags or (row + col) in anti_diags:
                continue
            cols.add(col)
            diags.add(row - col)
            anti_diags.add(row + col)
            board.append(col)
            _backtrack(row + 1)
            cols.remove(col)
            diags.remove(row - col)
            anti_diags.remove(row + col)
            board.pop()

    _backtrack(0)
    return solutions


def power_set(nums: List[int]) -> List[List[int]]:
    """
    Generate the power set of a list of integers using recursion.

    Parameters:
    -----------
    nums : List[int]
        The input list of integers.

    Returns:
    --------
    List[List[int]]
        A list containing all subsets of the input list.

    Raises:
    -------
    TypeError
        If nums is not a list of integers.
    """
    if not isinstance(nums, list):
        raise TypeError(f"Expected list, got {type(nums).__name__} instead.")
    if not all(isinstance(x, int) for x in nums):
        raise TypeError("All elements in the list must be integers.")

    def _backtrack(index: int, current: List[int], result: List[List[int]]) -> None:
        if index == len(nums):
            result.append(current.copy())
            return
        # Include nums[index]
        current.append(nums[index])
        _backtrack(index + 1, current, result)
        # Exclude nums[index]
        current.pop()
        _backtrack(index + 1, current, result)

    result: List[List[int]] = []
    _backtrack(0, [], result)
    return result


def matrix_power(matrix: torch.Tensor, exponent: int) -> torch.Tensor:
    """
    Compute the power of a square matrix using recursion and exponentiation by squaring.

    Parameters:
    -----------
    matrix : torch.Tensor
        A square matrix represented as a torch.Tensor.
    exponent : int
        The exponent to which the matrix is to be raised.

    Returns:
    --------
    torch.Tensor
        The resulting matrix after exponentiation.

    Raises:
    -------
    ValueError
        If the matrix is not square or exponent is negative.
    TypeError
        If matrix is not a torch.Tensor or exponent is not an integer.
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(matrix).__name__} instead.")
    if not isinstance(exponent, int):
        raise TypeError(f"Expected integer for exponent, got {type(exponent).__name__} instead.")
    if matrix.dim() != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError("Matrix must be square.")
    if exponent < 0:
        raise ValueError("Exponent must be non-negative.")

    def _matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)

    def _power(mat: torch.Tensor, exp: int) -> torch.Tensor:
        if exp == 0:
            return torch.eye(mat.size(0), dtype=mat.dtype)
        if exp == 1:
            return mat
        half = _power(mat, exp // 2)
        half_squared = _matrix_multiply(half, half)
        if exp % 2 == 0:
            return half_squared
        else:
            return _matrix_multiply(half_squared, mat)

    return _power(matrix, exponent)


def deep_copy(obj: Any) -> Any:
    """
    Create a deep copy of a nested list using recursion.

    Parameters:
    -----------
    obj : Any
        The object to be deep-copied. Expected to be a nested list.

    Returns:
    --------
    Any
        A deep copy of the input object.

    Raises:
    -------
    TypeError
        If obj contains non-list elements that are mutable and not handled.
    """
    if isinstance(obj, list):
        return [deep_copy(element) for element in obj]
    else:
        return obj  # For immutable objects, return as is


def tower_of_hanoi(n: int, source: str, target: str, auxiliary: str) -> List[str]:
    """
    Solve the Tower of Hanoi puzzle and return the list of moves.

    Parameters:
    -----------
    n : int
        Number of disks.
    source : str
        The source peg.
    target : str
        The target peg.
    auxiliary : str
        The auxiliary peg.

    Returns:
    --------
    List[str]
        A list of moves represented as strings.

    Raises:
    -------
    ValueError
        If n is negative.
    TypeError
        If n is not an integer or pegs are not strings.
    """
    if not isinstance(n, int):
        raise TypeError(f"Expected integer for n, got {type(n).__name__} instead.")
    if not all(isinstance(peg, str) for peg in [source, target, auxiliary]):
        raise TypeError("Peg names must be strings.")
    if n < 0:
        raise ValueError("Number of disks cannot be negative.")

    moves: List[str] = []

    def _hanoi(num: int, src: str, tgt: str, aux: str) -> None:
        if num == 1:
            moves.append(f"Move disk 1 from {src} to {tgt}")
            return
        _hanoi(num - 1, src, aux, tgt)
        moves.append(f"Move disk {num} from {src} to {tgt}")
        _hanoi(num - 1, aux, tgt, src)

    _hanoi(n, source, target, auxiliary)
    return moves


def subsets_sum(nums: List[int], target: int) -> List[List[int]]:
    """
    Find all unique subsets of nums that sum up to the target using recursion.

    Parameters:
    -----------
    nums : List[int]
        The list of integers.
    target : int
        The target sum.

    Returns:
    --------
    List[List[int]]
        A list of subsets that sum to the target.

    Raises:
    -------
    TypeError
        If nums is not a list of integers or target is not an integer.
    """
    if not isinstance(nums, list):
        raise TypeError(f"Expected list for nums, got {type(nums).__name__} instead.")
    if not all(isinstance(x, int) for x in nums):
        raise TypeError("All elements in nums must be integers.")
    if not isinstance(target, int):
        raise TypeError(f"Expected integer for target, got {type(target).__name__} instead.")

    nums.sort()

    def _find_subsets(index: int, current: List[int], current_sum: int, result: List[List[int]]) -> None:
        if current_sum == target:
            result.append(current.copy())
            return
        if current_sum > target or index == len(nums):
            return
        for i in range(index, len(nums)):
            # Skip duplicates
            if i > index and nums[i] == nums[i - 1]:
                continue
            current.append(nums[i])
            _find_subsets(i + 1, current, current_sum + nums[i], result)
            current.pop()

    result: List[List[int]] = []
    _find_subsets(0, [], 0, result)
    return result


def merge_sort(arr: List[int]) -> List[int]:
    """
    Sort an array of integers using the merge sort algorithm.

    Parameters:
    -----------
    arr : List[int]
        The list of integers to sort.

    Returns:
    --------
    List[int]
        A new sorted list.

    Raises:
    -------
    TypeError
        If arr is not a list of integers.
    """
    if not isinstance(arr, list):
        raise TypeError(f"Expected list, got {type(arr).__name__} instead.")
    if not all(isinstance(x, int) for x in arr):
        raise TypeError("All elements in the list must be integers.")
    if len(arr) <= 1:
        return arr.copy()

    mid = len(arr) // 2
    left_sorted = merge_sort(arr[:mid])
    right_sorted = merge_sort(arr[mid:])

    return _merge(left_sorted, right_sorted)


def _merge(left: List[int], right: List[int]) -> List[int]:
    """
    Merge two sorted lists into one sorted list.

    Parameters:
    -----------
    left : List[int]
        The first sorted list.
    right : List[int]
        The second sorted list.

    Returns:
    --------
    List[int]
        The merged and sorted list.
    """
    merged: List[int] = []
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


def count_paths(grid: List[List[int]], row: int = 0, col: int = 0) -> int:
    """
    Count the number of unique paths from top-left to bottom-right in a grid using recursion.

    Parameters:
    -----------
    grid : List[List[int]]
        A 2D grid represented as a list of lists.
    row : int, optional
        Current row position, by default 0.
    col : int, optional
        Current column position, by default 0.

    Returns:
    --------
    int
        The number of unique paths.

    Raises:
    -------
    TypeError
        If grid is not a list of lists of integers.
    ValueError
        If grid is empty or not rectangular.
    """
    if not isinstance(grid, list) or not all(isinstance(r, list) for r in grid):
        raise TypeError("Grid must be a list of lists.")
    if not grid or not grid[0]:
        raise ValueError("Grid cannot be empty.")
    num_rows = len(grid)
    num_cols = len(grid[0])
    if any(len(row_) != num_cols for row_ in grid):
        raise ValueError("All rows in the grid must have the same number of columns.")

    def _count(r: int, c: int) -> int:
        if r == num_rows - 1 and c == num_cols - 1:
            return 1
        if r >= num_rows or c >= num_cols:
            return 0
        return _count(r + 1, c) + _count(r, c + 1)

    return _count(row, col)


def deep_recursive_function(depth: int) -> str:
    """
    A demonstration of a deep recursive function that builds a string based on recursion depth.

    Parameters:
    -----------
    depth : int
        The depth of recursion.

    Returns:
    --------
    str
        A string representing the recursion path.

    Raises:
    -------
    ValueError
        If depth is negative.
    TypeError
        If depth is not an integer.
    """
    if not isinstance(depth, int):
        raise TypeError(f"Expected integer for depth, got {type(depth).__name__} instead.")
    if depth < 0:
        raise ValueError("Depth cannot be negative.")

    if depth == 0:
        return "Base case reached."
    return f"Recursion depth {depth} -> " + deep_recursive_function(depth - 1)


if __name__ == "__main__":
    # Example usages and test cases
    try:
        print("Factorial of 5:", factorial(5))
        print("Fibonacci of 10:", fibonacci(10))
        unsorted = [3, 6, 8, 10, 1, 2, 1]
        print("Quicksort:", quicksort(unsorted))
        sorted_arr = quicksort(unsorted)
        print("Binary Search for 10:", binary_search(sorted_arr, 10))
        
        # Binary Tree Example
        root = BinaryTreeNode(1)
        root.left = BinaryTreeNode(2)
        root.right = BinaryTreeNode(3)
        root.left.left = BinaryTreeNode(4)
        root.left.right = BinaryTreeNode(5)
        print("In-order Traversal:", inorder_traversal(root))
        
        # N-Queens
        n = 4
        print(f"Solutions for {n}-Queens:", solve_n_queens(n))
        
        # Power Set
        print("Power Set of [1, 2, 3]:", power_set([1, 2, 3]))
        
        # Matrix Power using torch
        mat = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        exponent = 3
        print(f"Matrix to the power of {exponent}:\n", matrix_power(mat, exponent))
        
        # Deep Copy
        original = [[1, 2], [3, 4]]
        copied = deep_copy(original)
        copied[0][0] = 99
        print("Original:", original)
        print("Copied:", copied)
        
        # Tower of Hanoi
        print("Tower of Hanoi Moves:", tower_of_hanoi(3, 'A', 'C', 'B'))
        
        # Subsets Sum
        print("Subsets summing to 5:", subsets_sum([1, 2, 3, 4, 5], 5))
        
        # Merge Sort
        print("Merge Sort:", merge_sort([12, 11, 13, 5, 6, 7]))
        
        # Count Paths in Grid
        grid_example = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        print("Number of unique paths:", count_paths(grid_example))
        
        # Deep Recursive Function
        print(deep_recursive_function(3))
        
    except Exception as e:
        print(f"An error occurred: {e}")