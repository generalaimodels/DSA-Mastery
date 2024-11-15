#!/usr/bin/env python3
"""
Backtracking Algorithms Module
==============================

This module provides implementations of various backtracking algorithms,
ranging from basic to advanced levels. Each algorithm is accompanied by
comprehensive error handling, type hints, and follows PEP-8 standards
for readability and maintainability.

Topics Covered:
- General Backtracking Framework
- N-Queens Problem
- Sudoku Solver
- Permutations and Combinations
- Subset Sum Problem
"""

from typing import List, Optional, Tuple
import sys


class Backtracking:
    """
    A class encapsulating various backtracking algorithms.
    """

    def __init__(self) -> None:
        """
        Initializes the Backtracking class.
        """
        pass

    def solve_n_queens(self, n: int) -> Optional[List[List[str]]]:
        """
        Solves the N-Queens problem and returns all distinct solutions.

        Args:
            n (int): The number of queens and the size of the chessboard.

        Returns:
            Optional[List[List[str]]]: A list of solutions, where each solution
            is represented as a list of strings. Returns None if n is invalid.
        """
        if n <= 0:
            print("Number of queens must be positive.", file=sys.stderr)
            return None

        solutions: List[List[str]] = []
        board: List[List[str]] = [["."] * n for _ in range(n)]

        def is_safe(row: int, col: int) -> bool:
            """
            Checks if a queen can be placed at the given position.

            Args:
                row (int): Row index.
                col (int): Column index.

            Returns:
                bool: True if safe, False otherwise.
            """
            for i in range(col):
                if board[row][i] == "Q":
                    return False

            for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
                if board[i][j] == "Q":
                    return False

            for i, j in zip(range(row, n, 1), range(col, -1, -1)):
                if board[i][j] == "Q":
                    return False

            return True

        def backtrack(col: int) -> None:
            """
            Utilizes backtracking to place queens column by column.

            Args:
                col (int): Current column to place the queen.
            """
            if col == n:
                solution = ["".join(row) for row in board]
                solutions.append(solution)
                return

            for row in range(n):
                if is_safe(row, col):
                    board[row][col] = "Q"
                    backtrack(col + 1)
                    board[row][col] = "."

        backtrack(0)
        return solutions

    def solve_sudoku(self, board: List[List[str]]) -> bool:
        """
        Solves a Sudoku puzzle using backtracking.

        Args:
            board (List[List[str]]): A 9x9 Sudoku board with empty cells represented by '.'.

        Returns:
            bool: True if the Sudoku puzzle is solved, False otherwise.
        """
        try:
            empty = self.find_empty(board)
            if not empty:
                return True  # Puzzle solved
            row, col = empty

            for num in map(str, range(1, 10)):
                if self.is_valid_sudoku(board, num, (row, col)):
                    board[row][col] = num

                    if self.solve_sudoku(board):
                        return True

                    board[row][col] = "."

            return False  # Trigger backtracking
        except IndexError as e:
            print(f"IndexError during Sudoku solving: {e}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            return False

    def find_empty(self, board: List[List[str]]) -> Optional[Tuple[int, int]]:
        """
        Finds an empty cell in the Sudoku board.

        Args:
            board (List[List[str]]): The Sudoku board.

        Returns:
            Optional[Tuple[int, int]]: The row and column of an empty cell, or None if full.
        """
        for i in range(9):
            for j in range(9):
                if board[i][j] == ".":
                    return i, j
        return None

    def is_valid_sudoku(self, board: List[List[str]], num: str, pos: Tuple[int, int]) -> bool:
        """
        Validates if placing a number at a given position is valid.

        Args:
            board (List[List[str]]): The Sudoku board.
            num (str): The number to place.
            pos (Tuple[int, int]): The (row, col) position.

        Returns:
            bool: True if valid, False otherwise.
        """
        row, col = pos

        # Check row
        if any(board[row][i] == num for i in range(9) if i != col):
            return False

        # Check column
        if any(board[i][col] == num for i in range(9) if i != row):
            return False

        # Check 3x3 grid
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] == num and (i, j) != pos:
                    return False

        return True

    def generate_permutations(self, nums: List[int]) -> List[List[int]]:
        """
        Generates all possible permutations of a list of numbers.

        Args:
            nums (List[int]): The list of numbers.

        Returns:
            List[List[int]]: A list of all possible permutations.
        """
        results: List[List[int]] = []
        n = len(nums)
        used = [False] * n
        permutation: List[int] = []

        def backtrack() -> None:
            if len(permutation) == n:
                results.append(permutation.copy())
                return
            for i in range(n):
                if not used[i]:
                    used[i] = True
                    permutation.append(nums[i])
                    backtrack()
                    permutation.pop()
                    used[i] = False

        backtrack()
        return results

    def generate_combinations(self, n: int, k: int) -> List[List[int]]:
        """
        Generates all possible combinations of k numbers out of 1 ... n.

        Args:
            n (int): The range of numbers from 1 to n.
            k (int): The number of elements in each combination.

        Returns:
            List[List[int]]: A list of all possible combinations.
        """
        results: List[List[int]] = []
        combination: List[int] = []

        def backtrack(start: int) -> None:
            if len(combination) == k:
                results.append(combination.copy())
                return
            for i in range(start, n + 1):
                combination.append(i)
                backtrack(i + 1)
                combination.pop()

        backtrack(1)
        return results

    def subset_sum(self, nums: List[int], target: int) -> List[List[int]]:
        """
        Finds all unique subsets of nums that sum up to target.

        Args:
            nums (List[int]): The list of integers.
            target (int): The target sum.

        Returns:
            List[List[int]]: A list of all unique subsets that sum to target.
        """
        results: List[List[int]] = []
        nums.sort()

        def backtrack(start: int, subset: List[int], total: int) -> None:
            if total == target:
                results.append(subset.copy())
                return
            if total > target:
                return
            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i - 1]:
                    continue  # Skip duplicates
                subset.append(nums[i])
                backtrack(i + 1, subset, total + nums[i])
                subset.pop()

        backtrack(0, [], 0)
        return results

    # Additional advanced backtracking algorithms can be added here


def main() -> None:
    """
    Main function to demonstrate the backtracking algorithms.
    """
    backtracker = Backtracking()

    # N-Queens Problem
    n = 4
    print(f"Solving {n}-Queens Problem:")
    solutions = backtracker.solve_n_queens(n)
    if solutions:
        for idx, solution in enumerate(solutions, start=1):
            print(f"Solution {idx}:")
            for row in solution:
                print(row)
            print()
    else:
        print("No solutions found.\n")

    # Sudoku Solver
    sudoku_board = [
        ["5", "3", ".", ".", "7", ".", ".", ".", "."],
        ["6", ".", ".", "1", "9", "5", ".", ".", "."],
        [".", "9", "8", ".", ".", ".", ".", "6", "."],
        ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
        ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
        ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
        [".", "6", ".", ".", ".", ".", "2", "8", "."],
        [".", ".", ".", "4", "1", "9", ".", ".", "5"],
        [".", ".", ".", ".", "8", ".", ".", "7", "9"],
    ]
    print("Solving Sudoku:")
    if backtracker.solve_sudoku(sudoku_board):
        for row in sudoku_board:
            print(" ".join(row))
    else:
        print("No solution exists for the provided Sudoku puzzle.")
    print()

    # Permutations
    nums = [1, 2, 3]
    print(f"Generating permutations for {nums}:")
    permutations = backtracker.generate_permutations(nums)
    for perm in permutations:
        print(perm)
    print()

    # Combinations
    n_comb, k = 4, 2
    print(f"Generating combinations of {k} numbers out of 1 to {n_comb}:")
    combinations = backtracker.generate_combinations(n_comb, k)
    for comb in combinations:
        print(comb)
    print()

    # Subset Sum
    subset_nums, target_sum = [2, 3, 6, 7], 7
    print(f"Finding subsets of {subset_nums} that sum to {target_sum}:")
    subsets = backtracker.subset_sum(subset_nums, target_sum)
    for subset in subsets:
        print(subset)
    print()


if __name__ == "__main__":
    main()