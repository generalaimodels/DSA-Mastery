"""
Dynamic Programming Module
==========================

This module provides a comprehensive overview of Dynamic Programming (DP),
covering concepts from basic to advanced levels. It includes implementations
of classic DP problems with detailed explanations, adhering to PEP-8 standards
and utilizing Python's typing module for clarity and maintainability.


"""

from typing import List, Tuple
import torch  # Imported as per instructions, though not utilized in this module


class DynamicProgramming:
    """
    A class encapsulating various dynamic programming algorithms.
    """

    def __init__(self):
        pass

    @staticmethod
    def fibonacci(n: int) -> int:
        """
        Calculate the nth Fibonacci number using memoization.

        Args:
            n (int): The position in the Fibonacci sequence.

        Returns:
            int: The nth Fibonacci number.

        Raises:
            ValueError: If n is negative.
        """
        if n < 0:
            raise ValueError("Fibonacci number is not defined for negative integers.")

        memo = {}

        def fib(k: int) -> int:
            if k in memo:
                return memo[k]
            if k <= 1:
                memo[k] = k
            else:
                memo[k] = fib(k - 1) + fib(k - 2)
            return memo[k]

        return fib(n)

    @staticmethod
    def knapsack(
        weights: List[int], values: List[int], capacity: int
    ) -> Tuple[int, List[int]]:
        """
        Solve the 0/1 Knapsack problem using dynamic programming.

        Args:
            weights (List[int]): The weights of the items.
            values (List[int]): The values of the items.
            capacity (int): The maximum capacity of the knapsack.

        Returns:
            Tuple[int, List[int]]: The maximum value achievable and the list of item indices included.

        Raises:
            ValueError: If weights and values lists have different lengths or capacity is negative.
        """
        if len(weights) != len(values):
            raise ValueError("Weights and values must be of the same length.")
        if capacity < 0:
            raise ValueError("Capacity cannot be negative.")

        n = len(weights)
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

        # Build table dp[][] in bottom up manner
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(
                        values[i - 1] + dp[i - 1][w - weights[i - 1]],
                        dp[i - 1][w],
                    )
                else:
                    dp[i][w] = dp[i - 1][w]

        # Trace back to find the items to include
        w = capacity
        selected_items = []
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected_items.append(i - 1)
                w -= weights[i - 1]

        return dp[n][capacity], selected_items[::-1]

    @staticmethod
    def longest_common_subsequence(text1: str, text2: str) -> int:
        """
        Find the length of the longest common subsequence between two strings.

        Args:
            text1 (str): The first string.
            text2 (str): The second string.

        Returns:
            int: The length of the longest common subsequence.

        Raises:
            ValueError: If either text1 or text2 is empty.
        """
        if not text1 or not text2:
            raise ValueError("Input strings must be non-empty.")

        m, n = len(text1), len(text2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    @staticmethod
    def edit_distance(word1: str, word2: str) -> int:
        """
        Calculate the minimum number of operations required to convert word1 to word2.

        Operations include insertions, deletions, or substitutions.

        Args:
            word1 (str): The source word.
            word2 (str): The target word.

        Returns:
            int: The minimum number of operations.

        Raises:
            ValueError: If either word1 or word2 is None.
        """
        if word1 is None or word2 is None:
            raise ValueError("Input words must not be None.")

        m, n = len(word1), len(word2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],    # Deletion
                        dp[i][j - 1],    # Insertion
                        dp[i - 1][j - 1] # Substitution
                    )

        return dp[m][n]

    @staticmethod
    def coin_change(coins: List[int], amount: int) -> int:
        """
        Determine the fewest number of coins needed to make up a given amount.

        Args:
            coins (List[int]): Available coin denominations.
            amount (int): The target amount.

        Returns:
            int: The minimum number of coins, or -1 if not possible.

        Raises:
            ValueError: If amount is negative or coins list is empty.
        """
        if amount < 0:
            raise ValueError("Amount cannot be negative.")
        if not coins:
            raise ValueError("Coins list cannot be empty.")

        dp = [float('inf')] * (amount + 1)
        dp[0] = 0

        for coin in coins:
            for x in range(coin, amount + 1):
                if dp[x - coin] + 1 < dp[x]:
                    dp[x] = dp[x - coin] + 1

        return dp[amount] if dp[amount] != float('inf') else -1

    @staticmethod
    def unique_paths(m: int, n: int) -> int:
        """
        Calculate the number of unique paths in an m x n grid from the top-left corner to the bottom-right corner.

        Args:
            m (int): Number of rows.
            n (int): Number of columns.

        Returns:
            int: The number of unique paths.

        Raises:
            ValueError: If m or n is not positive.
        """
        if m <= 0 or n <= 0:
            raise ValueError("Grid dimensions must be positive integers.")

        dp = [[1] * n for _ in range(m)]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[m - 1][n - 1]

    @staticmethod
    def max_subarray(nums: List[int]) -> int:
        """
        Find the contiguous subarray with the largest sum.

        Args:
            nums (List[int]): The input array of integers.

        Returns:
            int: The largest sum.

        Raises:
            ValueError: If the input list is empty.
        """
        if not nums:
            raise ValueError("Input list must not be empty.")

        max_current = max_global = nums[0]
        for num in nums[1:]:
            max_current = max(num, max_current + num)
            max_global = max(max_global, max_current)

        return max_global

    @staticmethod
    def partition_equal_subset(nums: List[int]) -> bool:
        """
        Determine if a list can be partitioned into two subsets with equal sums.

        Args:
            nums (List[int]): The list of positive integers.

        Returns:
            bool: True if possible, False otherwise.

        Raises:
            ValueError: If the list is empty.
        """
        if not nums:
            raise ValueError("Input list must not be empty.")

        total = sum(nums)
        if total % 2 != 0:
            return False

        target = total // 2
        dp = [False] * (target + 1)
        dp[0] = True

        for num in nums:
            for i in range(target, num - 1, -1):
                dp[i] = dp[i] or dp[i - num]

        return dp[target]

    @staticmethod
    def minimum_triangle_path(triangle: List[List[int]]) -> int:
        """
        Find the minimum path sum from top to bottom in a triangle.

        Args:
            triangle (List[List[int]]): The triangle represented as a list of lists.

        Returns:
            int: The minimum path sum.

        Raises:
            ValueError: If the triangle is empty or malformed.
        """
        if not triangle:
            raise ValueError("Triangle must not be empty.")

        n = len(triangle)
        dp = triangle[-1].copy()

        for i in range(n - 2, -1, -1):
            for j in range(len(triangle[i])):
                dp[j] = triangle[i][j] + min(dp[j], dp[j + 1])

        return dp[0]

    @staticmethod
    def longest_increasing_subsequence(nums: List[int]) -> int:
        """
        Find the length of the longest increasing subsequence.

        Args:
            nums (List[int]): The input array of integers.

        Returns:
            int: The length of the longest increasing subsequence.

        Raises:
            ValueError: If the input list is empty.
        """
        if not nums:
            raise ValueError("Input list must not be empty.")

        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)

    @staticmethod
    def matrix_chain_order(p: List[int]) -> int:
        """
        Determine the most efficient way to multiply a given sequence of matrices.

        Args:
            p (List[int]): The dimensions of the matrices such that matrix i has dimensions p[i-1] x p[i].

        Returns:
            int: The minimum number of scalar multiplications needed.

        Raises:
            ValueError: If the list p has fewer than two elements.
        """
        if len(p) < 2:
            raise ValueError("List p must contain at least two dimensions.")

        n = len(p) - 1
        dp = [[0 for _ in range(n)] for _ in range(n)]

        for chain_length in range(2, n + 1):
            for i in range(n - chain_length + 1):
                j = i + chain_length - 1
                dp[i][j] = float('inf')
                for k in range(i, j):
                    cost = (
                        dp[i][k]
                        + dp[k + 1][j]
                        + p[i] * p[k + 1] * p[j + 1]
                    )
                    if cost < dp[i][j]:
                        dp[i][j] = cost

        return dp[0][n - 1]

    @staticmethod
    def coin_change_combinations(coins: List[int], amount: int) -> int:
        """
        Compute the number of combinations that make up a given amount.

        Args:
            coins (List[int]): The coin denominations.
            amount (int): The target amount.

        Returns:
            int: The number of combinations.

        Raises:
            ValueError: If amount is negative or coins list is empty.
        """
        if amount < 0:
            raise ValueError("Amount cannot be negative.")
        if not coins:
            raise ValueError("Coins list cannot be empty.")

        dp = [0] * (amount + 1)
        dp[0] = 1

        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] += dp[x - coin]

        return dp[amount]

    @staticmethod
    def minimum_cost_path(grid: List[List[int]]) -> int:
        """
        Find the minimum cost path from the top-left to the bottom-right of a grid.

        Args:
            grid (List[List[int]]): The grid representing cost at each cell.

        Returns:
            int: The minimum cost path.

        Raises:
            ValueError: If the grid is empty or not rectangular.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must not be empty and must be rectangular.")

        m, n = len(grid), len(grid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]
        dp[0][0] = grid[0][0]

        # Initialize first row
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] + grid[0][j]

        # Initialize first column
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + grid[i][0]

        # Populate the rest of the dp table
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])

        return dp[-1][-1]

    @staticmethod
    def longest_palindromic_subsequence(s: str) -> int:
        """
        Find the length of the longest palindromic subsequence in a string.

        Args:
            s (str): The input string.

        Returns:
            int: The length of the longest palindromic subsequence.

        Raises:
            ValueError: If the input string is empty.
        """
        if not s:
            raise ValueError("Input string must not be empty.")

        n = len(s)
        dp = [[0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            dp[i][i] = 1

        for cl in range(2, n + 1):
            for i in range(n - cl + 1):
                j = i + cl - 1
                if s[i] == s[j]:
                    dp[i][j] = 2 + dp[i + 1][j - 1] if cl > 2 else 2
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i + 1][j])

        return dp[0][n - 1]

    @staticmethod
    def number_of_decodings(s: str) -> int:
        """
        Calculate the number of ways to decode a string of digits into letters.

        Args:
            s (str): The encoded string.

        Returns:
            int: The number of decoding ways.

        Raises:
            ValueError: If the input string contains invalid characters.
        """
        if not s:
            return 0
        if any(c < '0' or c > '9' for c in s):
            raise ValueError("Input string must contain only digits.")

        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 1

        for i in range(1, n + 1):
            if s[i - 1] != '0':
                dp[i] += dp[i - 1]
            if i > 1 and '10' <= s[i - 2:i] <= '26':
                dp[i] += dp[i - 2]

        return dp[n]

    @staticmethod
    def burst_balloons(nums: List[int]) -> int:
        """
        Find the maximum coins you can collect by bursting balloons wisely.

        Args:
            nums (List[int]): The list of balloon numbers.

        Returns:
            int: The maximum coins that can be collected.

        Raises:
            ValueError: If the input list is empty.
        """
        if not nums:
            raise ValueError("Input list must not be empty.")

        nums = [1] + nums + [1]
        n = len(nums)
        dp = [[0] * n for _ in range(n)]

        for length in range(2, n):
            for left in range(0, n - length):
                right = left + length
                for i in range(left + 1, right):
                    coins = nums[left] * nums[i] * nums[right]
                    total = coins + dp[left][i] + dp[i][right]
                    if total > dp[left][right]:
                        dp[left][right] = total

        return dp[0][n - 1]

    @staticmethod
    def unique_binary_search_trees(n: int) -> int:
        """
        Calculate the number of unique Binary Search Trees (BSTs) that can be formed with n nodes.

        Args:
            n (int): The number of nodes.

        Returns:
            int: The number of unique BSTs.

        Raises:
            ValueError: If n is negative.
        """
        if n < 0:
            raise ValueError("n must be a non-negative integer.")

        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1

        for nodes in range(2, n + 1):
            for root in range(1, nodes + 1):
                dp[nodes] += dp[root - 1] * dp[nodes - root]

        return dp[n]

    @staticmethod
    def integer_break(n: int) -> int:
        """
        Break an integer into the sum of at least two positive integers and maximize the product of those integers.

        Args:
            n (int): The integer to break.

        Returns:
            int: The maximum product.

        Raises:
            ValueError: If n is less than 2.
        """
        if n < 2:
            raise ValueError("n must be at least 2.")

        dp = [0] * (n + 1)
        dp[1] = 1

        for i in range(2, n + 1):
            for j in range(1, i):
                dp[i] = max(dp[i], j * (i - j), j * dp[i - j])

        return dp[n]

    @staticmethod
    def paint_fence(k: int, n: int) -> int:
        """
        Calculate the number of ways to paint a fence with n posts using k colors without painting three or more consecutive posts with the same color.

        Args:
            k (int): Number of colors.
            n (int): Number of posts.

        Returns:
            int: The number of ways to paint the fence.

        Raises:
            ValueError: If k or n is negative.
        """
        if k <= 0 or n < 0:
            raise ValueError("k must be positive and n must be non-negative.")

        if n == 0:
            return 0
        if n == 1:
            return k

        dp = [0] * (n + 1)
        dp[1] = k
        dp[2] = k * k

        for i in range(3, n + 1):
            dp[i] = (k - 1) * (dp[i - 1] + dp[i - 2])

        return dp[n]

    @staticmethod
    def palindrome_partition(s: str) -> int:
        """
        Determine the minimum number of cuts needed to partition a string such that each substring is a palindrome.

        Args:
            s (str): The input string.

        Returns:
            int: The minimum number of cuts.

        Raises:
            ValueError: If the input string is empty.
        """
        if not s:
            raise ValueError("Input string must not be empty.")

        n = len(s)
        dp = [0] * n
        palindrome = [[False] * n for _ in range(n)]

        for i in range(n):
            min_cut = i  # Maximum cuts
            for j in range(i + 1):
                if s[j] == s[i] and (j + 1 > i - 1 or palindrome[j + 1][i - 1]):
                    palindrome[j][i] = True
                    min_cut = 0 if j == 0 else min(min_cut, dp[j - 1] + 1)
            dp[i] = min_cut

        return dp[-1]