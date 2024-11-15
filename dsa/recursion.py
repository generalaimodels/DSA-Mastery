"""
Recursion in Python: From Basics to Advanced Techniques
=======================================================

This module provides a comprehensive exploration of recursion in Python, covering fundamental concepts to advanced applications. Each section includes detailed explanations, type hints, error handling, and efficient implementations adhering to PEP-8 standards.

Contents:
---------
1. Introduction to Recursion
2. Basic Recursive Functions
    - Factorial Calculation
    - Fibonacci Sequence
3. Recursive Data Structures
    - Linked Lists
    - Binary Trees
4. Advanced Recursion Techniques
    - Memoization
    - Tail Recursion Optimization
5. Practical Applications
    - Sorting Algorithms (Quick Sort)
    - Backtracking (N-Queens Problem)
6. Iterative vs Recursive Solutions
7. Conclusion

Author: Generalmodelai-agent
Date: 2024-10-15
"""

from typing import Optional, List, Any, Dict, Tuple
import sys


class RecursionError(Exception):
    """Custom exception for recursion-related errors."""
    pass


# 1. Introduction to Recursion
# ----------------------------
# Recursion is a programming technique where a function calls itself to solve smaller instances of a problem.
# It is essential for solving problems that can be broken down into similar subproblems.

# 2. Basic Recursive Functions
# -----------------------------


def factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer n using recursion.

    Args:
        n (int): A non-negative integer whose factorial is to be computed.

    Returns:
        int: The factorial of the given number.

    Raises:
        RecursionError: If n is negative.
        ValueError: If n is not an integer.
    """
    if not isinstance(n, int):
        raise ValueError("Factorial is only defined for integers.")
    if n < 0:
        raise RecursionError("Factorial is not defined for negative integers.")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)


def fibonacci(n: int) -> int:
    """
    Compute the nth Fibonacci number using recursion.

    Args:
        n (int): The position in the Fibonacci sequence.

    Returns:
        int: The nth Fibonacci number.

    Raises:
        RecursionError: If n is negative.
        ValueError: If n is not an integer.
    """
    if not isinstance(n, int):
        raise ValueError("Fibonacci number is only defined for integers.")
    if n < 0:
        raise RecursionError("Fibonacci number is not defined for negative integers.")
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


# 3. Recursive Data Structures
# ----------------------------

class ListNode:
    """
    A node in a singly linked list.

    Attributes:
        value (Any): The value stored in the node.
        next (Optional['ListNode']): The reference to the next node.
    """

    def __init__(self, value: Any, next: Optional['ListNode'] = None):
        self.value = value
        self.next = next

    def __repr__(self) -> str:
        return f"ListNode({self.value})"


class BinaryTreeNode:
    """
    A node in a binary tree.

    Attributes:
        value (Any): The value stored in the node.
        left (Optional['BinaryTreeNode']): The left child node.
        right (Optional['BinaryTreeNode']): The right child node.
    """

    def __init__(self, value: Any, left: Optional['BinaryTreeNode'] = None,
                 right: Optional['BinaryTreeNode'] = None):
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"BinaryTreeNode({self.value})"


def sum_linked_list(node: Optional[ListNode]) -> int:
    """
    Recursively sum all values in a linked list.

    Args:
        node (Optional[ListNode]): The head node of the linked list.

    Returns:
        int: The sum of all node values.
    """
    if node is None:
        return 0
    return node.value + sum_linked_list(node.next)


def inorder_traversal(node: Optional[BinaryTreeNode]) -> List[Any]:
    """
    Perform in-order traversal of a binary tree.

    Args:
        node (Optional[BinaryTreeNode]): The root node of the binary tree.

    Returns:
        List[Any]: A list of values in in-order sequence.
    """
    if node is None:
        return []
    return inorder_traversal(node.left) + [node.value] + inorder_traversal(node.right)


# 4. Advanced Recursion Techniques
# ---------------------------------

from functools import lru_cache


@lru_cache(maxsize=None)
def memoized_fibonacci(n: int) -> int:
    """
    Compute the nth Fibonacci number using memoization to optimize recursion.

    Args:
        n (int): The position in the Fibonacci sequence.

    Returns:
        int: The nth Fibonacci number.

    Raises:
        RecursionError: If n is negative.
        ValueError: If n is not an integer.
    """
    if not isinstance(n, int):
        raise ValueError("Fibonacci number is only defined for integers.")
    if n < 0:
        raise RecursionError("Fibonacci number is not defined for negative integers.")
    if n == 0:
        return 0
    if n == 1:
        return 1
    return memoized_fibonacci(n - 1) + memoized_fibonacci(n - 2)


def tail_recursive_factorial(n: int, accumulator: int = 1) -> int:
    """
    Calculate the factorial of a non-negative integer n using tail recursion.

    Args:
        n (int): A non-negative integer whose factorial is to be computed.
        accumulator (int): The accumulated result (used in tail recursion).

    Returns:
        int: The factorial of the given number.

    Raises:
        RecursionError: If n is negative.
        ValueError: If n is not an integer.
    """
    if not isinstance(n, int):
        raise ValueError("Factorial is only defined for integers.")
    if n < 0:
        raise RecursionError("Factorial is not defined for negative integers.")
    if n == 0:
        return accumulator
    return tail_recursive_factorial(n - 1, n * accumulator)


# 5. Practical Applications
# -------------------------

def quick_sort(arr: List[int]) -> List[int]:
    """
    Sort a list of integers using the Quick Sort algorithm implemented recursively.

    Args:
        arr (List[int]): The list of integers to sort.

    Returns:
        List[int]: A new sorted list.
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


def solve_n_queens(n: int) -> List[List[str]]:
    """
    Solve the N-Queens problem using backtracking with recursion.

    Args:
        n (int): The number of queens and the size of the board (n x n).

    Returns:
        List[List[str]]: A list of all possible solutions, each represented by a list of strings.
    """
    def backtrack(row: int, diagonals: set, anti_diagonals: set,
                  cols: set, state: List[int]) -> None:
        if row == n:
            board = []
            for i in state:
                line = ['.'] * n
                line[i] = 'Q'
                board.append("".join(line))
            solutions.append(board)
            return
        for col in range(n):
            diag = row - col
            anti_diag = row + col
            if (col in cols or diag in diagonals or
                    anti_diag in anti_diagonals):
                continue
            cols.add(col)
            diagonals.add(diag)
            anti_diagonals.add(anti_diag)
            state.append(col)
            backtrack(row + 1, diagonals, anti_diagonals, cols, state)
            cols.remove(col)
            diagonals.remove(diag)
            anti_diagonals.remove(anti_diag)
            state.pop()

    solutions: List[List[str]] = []
    backtrack(0, set(), set(), set(), [])
    return solutions


# 6. Iterative vs Recursive Solutions
# ------------------------------------

def iterative_factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer n using an iterative approach.

    Args:
        n (int): A non-negative integer whose factorial is to be computed.

    Returns:
        int: The factorial of the given number.
    """
    if not isinstance(n, int):
        raise ValueError("Factorial is only defined for integers.")
    if n < 0:
        raise RecursionError("Factorial is not defined for negative integers.")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


# 7. Conclusion
# -------------

def main() -> None:
    """
    Main function to demonstrate various recursion implementations.
    """
    try:
        print("Factorial of 5:", factorial(5))
        print("Fibonacci of 10:", fibonacci(10))
        print("Memoized Fibonacci of 50:", memoized_fibonacci(50))
        print("Tail Recursive Factorial of 6:", tail_recursive_factorial(6))
        print("Quick Sort Example:", quick_sort([3, 6, 8, 10, 1, 2, 1]))
        
        # N-Queens Solution for n=4
        solutions = solve_n_queens(4)
        print(f"N-Queens Solutions for n=4: {len(solutions)} found")
        for solution in solutions:
            for row in solution:
                print(row)
            print()

        # Comparing Iterative and Recursive Factorial
        print("Iterative Factorial of 5:", iterative_factorial(5))

    except (RecursionError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()