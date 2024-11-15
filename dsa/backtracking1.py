"""
backtracking.py

A comprehensive exploration of Backtracking algorithms in Python, ranging from basic
implementations to advanced, professional-level solutions. This module adheres to
PEP-8 standards, utilizes type hints for clarity, and includes robust error handling
to ensure reliability and scalability.

Topics Covered:
1. Introduction to Backtracking
2. Permutations Generation
3. N-Queens Problem
4. Sudoku Solver
5. Subset Sum Problem


"""

from typing import List, Optional, Generator


class BacktrackingError(Exception):
    """Custom exception class for Backtracking-related errors."""
    pass


def backtrack(
    solution: List,
    options: List,
    is_valid: callable,
    is_solution: callable,
    process_solution: callable
) -> None:
    """
    A generic backtracking algorithm.

    :param solution: Current partial solution.
    :param options: Available options at the current step.
    :param is_valid: Function to check if the current partial solution is valid.
    :param is_solution: Function to check if the current partial solution is a complete solution.
    :param process_solution: Function to process a complete solution.
    """
    if is_solution(solution):
        process_solution(solution)
    else:
        for option in options:
            solution.append(option)
            if is_valid(solution):
                backtrack(solution, options, is_valid, is_solution, process_solution)
            solution.pop()


def generate_permutations(nums: List[int]) -> Generator[List[int], None, None]:
    """
    Generate all possible permutations of a list of numbers using backtracking.

    :param nums: List of integers to permute.
    :yield: Generator yielding each permutation as a list.
    """

    def is_valid(curr_solution: List[int]) -> bool:
        # Ensure no duplicate elements
        return len(curr_solution) == len(set(curr_solution))

    def is_solution(curr_solution: List[int]) -> bool:
        return len(curr_solution) == len(nums)

    def process_solution(curr_solution: List[int]) -> None:
        yield_list.append(curr_solution.copy())

    yield_list: List[List[int]] = []
    backtrack([], nums, is_valid, is_solution, process_solution)
    for perm in yield_list:
        yield perm


def solve_n_queens(n: int) -> List[List[str]]:
    """
    Solve the N-Queens problem and return all distinct solutions.

    Each solution contains a distinct board configuration of the N-Queens' placement.

    :param n: Number of queens and size of the board (n x n).
    :return: A list of solutions, where each solution is represented as a list of strings.
    :raises BacktrackingError: If input n is invalid.
    """

    if n <= 0:
        raise BacktrackingError("Number of queens must be a positive integer.")

    solutions: List[List[str]] = []
    board: List[str] = ["." * n for _ in range(n)]

    def is_valid(row: int, col: int, queens: List[int]) -> bool:
        for r, c in enumerate(queens):
            if c == col or abs(r - row) == abs(c - col):
                return False
        return True

    def solve(row: int, queens: List[int]) -> None:
        if row == n:
            solution = []
            for q in queens:
                line = '.' * q + 'Q' + '.' * (n - q - 1)
                solution.append(line)
            solutions.append(solution)
            return
        for col in range(n):
            if is_valid(row, col, queens):
                queens.append(col)
                solve(row + 1, queens)
                queens.pop()

    solve(0, [])
    return solutions


def solve_sudoku(board: List[List[str]]) -> bool:
    """
    Solve a Sudoku puzzle using backtracking.

    :param board: 9x9 Sudoku board with empty cells represented by '.'.
    :return: True if the Sudoku is solvable, False otherwise.
    :raises BacktrackingError: If the input board is invalid.
    """

    if not board or len(board) != 9 or any(len(row) != 9 for row in board):
        raise BacktrackingError("Invalid Sudoku board size.")

    def is_valid(r: int, c: int, char: str) -> bool:
        for i in range(9):
            if board[r][i] == char or board[i][c] == char:
                return False
            if board[3 * (r // 3) + i // 3][3 * (c // 3) + i % 3] == char:
                return False
        return True

    def solve() -> bool:
        for r in range(9):
            for c in range(9):
                if board[r][c] == '.':
                    for char in '123456789':
                        if is_valid(r, c, char):
                            board[r][c] = char
                            if solve():
                                return True
                            board[r][c] = '.'
                    return False
        return True

    return solve()


def subset_sum(nums: List[int], target: int) -> List[List[int]]:
    """
    Find all unique subsets in nums that sum up to the target using backtracking.

    :param nums: List of integers.
    :param target: Target sum.
    :return: A list of unique subsets that sum to the target.
    :raises BacktrackingError: If nums is empty or target is not an integer.
    """

    if not isinstance(target, int):
        raise BacktrackingError("Target must be an integer.")
    if not nums:
        return []

    nums.sort()
    results: List[List[int]] = []

    def backtrack_subset(start: int, current: List[int], current_sum: int) -> None:
        if current_sum == target:
            results.append(current.copy())
            return
        if current_sum > target:
            return
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            current.append(nums[i])
            backtrack_subset(i + 1, current, current_sum + nums[i])
            current.pop()

    backtrack_subset(0, [], 0)
    return results


def main() -> None:
    """
    Main function to demonstrate backtracking algorithms.
    """

    # Example 1: Permutations
    print("Permutations of [1, 2, 3]:")
    for perm in generate_permutations([1, 2, 3]):
        print(perm)
    print()

    # Example 2: N-Queens
    n = 4
    print(f"Solutions for {n}-Queens problem:")
    try:
        queens_solutions = solve_n_queens(n)
        for solution in queens_solutions:
            for row in solution:
                print(row)
            print()
    except BacktrackingError as e:
        print(f"Error: {e}")
    print()

    # Example 3: Sudoku Solver
    sudoku_board = [
        ['5', '3', '.', '.', '7', '.', '.', '.', '.'],
        ['6', '.', '.', '1', '9', '5', '.', '.', '.'],
        ['.', '9', '8', '.', '.', '.', '.', '6', '.'],
        ['8', '.', '.', '.', '6', '.', '.', '.', '3'],
        ['4', '.', '.', '8', '.', '3', '.', '.', '1'],
        ['7', '.', '.', '.', '2', '.', '.', '.', '6'],
        ['.', '6', '.', '.', '.', '.', '2', '8', '.'],
        ['.', '.', '.', '4', '1', '9', '.', '.', '5'],
        ['.', '.', '.', '.', '8', '.', '.', '7', '9']
    ]
    print("Sudoku Solver:")
    try:
        if solve_sudoku(sudoku_board):
            for row in sudoku_board:
                print(" ".join(row))
        else:
            print("No solution exists.")
    except BacktrackingError as e:
        print(f"Error: {e}")
    print()

    # Example 4: Subset Sum
    nums = [2, 3, 6, 7]
    target = 7
    print(f"Subset Sum solutions for target {target}:")
    try:
        subsets = subset_sum(nums, target)
        for subset in subsets:
            print(subset)
    except BacktrackingError as e:
        print(f"Error: {e}")
    print()


if __name__ == "__main__":
    main()