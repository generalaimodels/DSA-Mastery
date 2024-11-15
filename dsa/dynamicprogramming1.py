"""
Dynamic Programming Module
==========================

This module provides a comprehensive overview of Dynamic Programming (DP),
covering fundamental to advanced topics. It includes various DP techniques,
examples, and implementations to illustrate concepts effectively.

Dynamic Programming is a method for solving complex problems by breaking them
down into simpler subproblems. It is applicable where the problem can be divided
into overlapping subproblems that can be solved independently, and their solutions
can be cached for future reference.

This module adheres to PEP-8 standards, utilizes type hints for clarity, and includes
robust error handling to ensure reliability and maintainability.


"""

from typing import List, Dict, Tuple


class DynamicProgramming:
    """
    A class encapsulating various Dynamic Programming algorithms and techniques.
    """

    def fibonacci_recursive(self, n: int) -> int:
        """
        Calculate the nth Fibonacci number using a recursive approach.

        Parameters:
            n (int): The position in the Fibonacci sequence.

        Returns:
            int: The nth Fibonacci number.

        Raises:
            ValueError: If n is negative.
        """
        if n < 0:
            raise ValueError("Fibonacci number is not defined for negative integers.")

        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            return self.fibonacci_recursive(n - 1) + self.fibonacci_recursive(n - 2)

    def fibonacci_memoization(self, n: int, memo: Dict[int, int] = None) -> int:
        """
        Calculate the nth Fibonacci number using memoization.

        Parameters:
            n (int): The position in the Fibonacci sequence.
            memo (Dict[int, int], optional): A dictionary to store previously computed values.

        Returns:
            int: The nth Fibonacci number.

        Raises:
            ValueError: If n is negative.
        """
        if memo is None:
            memo = {}

        if n < 0:
            raise ValueError("Fibonacci number is not defined for negative integers.")

        if n in memo:
            return memo[n]

        if n == 0:
            memo[0] = 0
        elif n == 1:
            memo[1] = 1
        else:
            memo[n] = self.fibonacci_memoization(n - 1, memo) + self.fibonacci_memoization(n - 2, memo)

        return memo[n]

    def fibonacci_bottom_up(self, n: int) -> int:
        """
        Calculate the nth Fibonacci number using a bottom-up dynamic programming approach.

        Parameters:
            n (int): The position in the Fibonacci sequence.

        Returns:
            int: The nth Fibonacci number.

        Raises:
            ValueError: If n is negative.
        """
        if n < 0:
            raise ValueError("Fibonacci number is not defined for negative integers.")

        if n == 0:
            return 0
        elif n == 1:
            return 1

        fib_table = [0] * (n + 1)
        fib_table[0] = 0
        fib_table[1] = 1

        for i in range(2, n + 1):
            fib_table[i] = fib_table[i - 1] + fib_table[i - 2]

        return fib_table[n]

    def knapsack_0_1(self, weights: List[int], values: List[int], capacity: int) -> int:
        """
        Solve the 0/1 Knapsack problem using dynamic programming.

        Parameters:
            weights (List[int]): The weights of the items.
            values (List[int]): The values of the items.
            capacity (int): The maximum capacity of the knapsack.

        Returns:
            int: The maximum value achievable within the given capacity.

        Raises:
            ValueError: If lengths of weights and values do not match or if capacity is negative.
        """
        if len(weights) != len(values):
            raise ValueError("Weights and values must be of the same length.")
        if capacity < 0:
            raise ValueError("Capacity cannot be negative.")

        n = len(weights)
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(
                        values[i - 1] + dp[i - 1][w - weights[i - 1]],
                        dp[i - 1][w]
                    )
                else:
                    dp[i][w] = dp[i - 1][w]

        return dp[n][capacity]

    def longest_common_subsequence(self, text1: str, text2: str) -> int:
        """
        Find the length of the longest common subsequence between two strings.

        Parameters:
            text1 (str): The first string.
            text2 (str): The second string.

        Returns:
            int: The length of the longest common subsequence.

        Raises:
            ValueError: If either input string is empty.
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

    def edit_distance(self, word1: str, word2: str) -> int:
        """
        Compute the minimum number of operations required to convert word1 to word2.

        Operations include insertions, deletions, or substitutions.

        Parameters:
            word1 (str): The source word.
            word2 (str): The target word.

        Returns:
            int: The minimum number of operations.

        Raises:
            ValueError: If either input word is None.
        """
        if word1 is None or word2 is None:
            raise ValueError("Input words cannot be None.")

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

    def coin_change(self, coins: List[int], amount: int) -> int:
        """
        Determine the fewest number of coins needed to make up a given amount.

        Parameters:
            coins (List[int]): The denominations of the coins.
            amount (int): The total amount.

        Returns:
            int: The minimum number of coins needed, or -1 if not possible.

        Raises:
            ValueError: If amount is negative.
        """
        if amount < 0:
            raise ValueError("Amount cannot be negative.")
        if not coins:
            return -1

        dp = [amount + 1] * (amount + 1)
        dp[0] = 0

        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)

        return dp[amount] if dp[amount] <= amount else -1

    def unique_paths(self, m: int, n: int) -> int:
        """
        Calculate the number of unique paths in an m x n grid from top-left to bottom-right.

        Parameters:
            m (int): The number of rows.
            n (int): The number of columns.

        Returns:
            int: The number of unique paths.

        Raises:
            ValueError: If m or n is not positive.
        """
        if m <= 0 or n <= 0:
            raise ValueError("Grid dimensions must be positive integers.")

        dp = [[1 for _ in range(n)] for _ in range(m)]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[m - 1][n - 1]

    def maximum_subarray(self, nums: List[int]) -> int:
        """
        Find the contiguous subarray with the largest sum.

        Parameters:
            nums (List[int]): The list of integers.

        Returns:
            int: The largest sum of the contiguous subarray.

        Raises:
            ValueError: If nums is empty.
        """
        if not nums:
            raise ValueError("The input list cannot be empty.")

        max_current = max_global = nums[0]

        for num in nums[1:]:
            max_current = max(num, max_current + num)
            max_global = max(max_global, max_current)

        return max_global

    def rod_cutting(self, prices: List[int], n: int) -> int:
        """
        Determine the maximum profit obtainable by cutting a rod of length n.

        Parameters:
            prices (List[int]): The prices of rods of different lengths.
            n (int): The length of the rod.

        Returns:
            int: The maximum profit.

        Raises:
            ValueError: If n is negative or prices list is empty.
        """
        if n < 0:
            raise ValueError("Rod length cannot be negative.")
        if not prices:
            raise ValueError("Prices list cannot be empty.")

        dp = [0] * (n + 1)

        for i in range(1, n + 1):
            max_val = float('-inf')
            for j in range(1, min(i, len(prices)) + 1):
                max_val = max(max_val, prices[j - 1] + dp[i - j])
            dp[i] = max_val

        return dp[n]

    def longest_increasing_subsequence(self, nums: List[int]) -> int:
        """
        Find the length of the longest increasing subsequence.

        Parameters:
            nums (List[int]): The list of integers.

        Returns:
            int: The length of the longest increasing subsequence.

        Raises:
            ValueError: If nums is empty.
        """
        if not nums:
            raise ValueError("The input list cannot be empty.")

        dp = [1] * len(nums)

        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)

    def matrix_chain_multiplication(self, matrices: List[int]) -> int:
        """
        Determine the minimum number of scalar multiplications needed to multiply a chain of matrices.

        Parameters:
            matrices (List[int]): The dimensions of the matrices. Length should be number of matrices +1.

        Returns:
            int: The minimum number of multiplications.

        Raises:
            ValueError: If the number of matrices is less than 2 or dimensions are incompatible.
        """
        if len(matrices) < 2:
            raise ValueError("At least two matrices are required.")

        n = len(matrices) - 1
        dp = [[0 for _ in range(n)] for _ in range(n)]

        for chain_length in range(2, n + 1):
            for i in range(n - chain_length + 1):
                j = i + chain_length - 1
                dp[i][j] = float('inf')
                for k in range(i, j):
                    cost = (
                        dp[i][k]
                        + dp[k + 1][j]
                        + matrices[i] * matrices[k + 1] * matrices[j + 1]
                    )
                    dp[i][j] = min(dp[i][j], cost)

        return dp[0][n - 1]

    def palindrome_partitioning(self, s: str) -> int:
        """
        Partition a string such that every substring is a palindrome with minimum cuts.

        Parameters:
            s (str): The input string.

        Returns:
            int: The minimum number of cuts required.

        Raises:
            ValueError: If the input string is empty.
        """
        if not s:
            raise ValueError("Input string cannot be empty.")

        n = len(s)
        dp = [0] * n
        palindrome = [[False] * n for _ in range(n)]

        for i in range(n):
            min_cuts = i
            for j in range(i + 1):
                if s[j] == s[i] and (i - j <= 1 or palindrome[j + 1][i - 1]):
                    palindrome[j][i] = True
                    min_cuts = 0 if j == 0 else min(min_cuts, dp[j - 1] + 1)
            dp[i] = min_cuts

        return dp[-1]

    def unique_binary_search_trees(self, n: int) -> int:
        """
        Count the number of unique binary search trees that can be formed with n distinct nodes.

        Parameters:
            n (int): The number of distinct nodes.

        Returns:
            int: The number of unique binary search trees.

        Raises:
            ValueError: If n is negative.
        """
        if n < 0:
            raise ValueError("Number of nodes cannot be negative.")

        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1

        for nodes in range(2, n + 1):
            for root in range(1, nodes + 1):
                dp[nodes] += dp[root - 1] * dp[nodes - root]

        return dp[n]

    def min_cost_climbing_stairs(self, cost: List[int]) -> int:
        """
        Find the minimum cost to reach the top of the floor.

        You can either start from the step with index 0, or the step with index 1.

        Parameters:
            cost (List[int]): The cost of each step.

        Returns:
            int: The minimum cost to reach the top.

        Raises:
            ValueError: If the cost list is empty.
        """
        if not cost:
            raise ValueError("Cost list cannot be empty.")

        n = len(cost)
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 0

        for i in range(2, n + 1):
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])

        return dp[n]

    def word_break(self, s: str, word_dict: List[str]) -> bool:
        """
        Determine if the string can be segmented into a space-separated sequence of one or more dictionary words.

        Parameters:
            s (str): The input string.
            word_dict (List[str]): The list of valid words.

        Returns:
            bool: True if the string can be segmented, False otherwise.

        Raises:
            ValueError: If the input string is empty.
        """
        if not s:
            raise ValueError("Input string cannot be empty.")
        if not word_dict:
            return False

        word_set = set(word_dict)
        dp = [False] * (len(s) + 1)
        dp[0] = True

        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break

        return dp[-1]

    def paint_house(self, costs: List[List[int]]) -> int:
        """
        Find the minimum cost to paint all houses such that no two adjacent houses have the same color.

        Parameters:
            costs (List[List[int]]): A list of costs where costs[i][j] is the cost of painting house i with color j.

        Returns:
            int: The minimum cost to paint all houses.

        Raises:
            ValueError: If the costs list is empty or improperly formatted.
        """
        if not costs or not all(len(color) == len(costs[0]) for color in costs):
            raise ValueError("Invalid costs list.")

        n = len(costs)
        k = len(costs[0])
        dp = [0] * k

        for j in range(k):
            dp[j] = costs[0][j]

        for i in range(1, n):
            new_dp = [0] * k
            for j in range(k):
                min_cost = float('inf')
                for l in range(k):
                    if l != j:
                        min_cost = min(min_cost, dp[l])
                new_dp[j] = costs[i][j] + min_cost
            dp = new_dp

        return min(dp)

    def house_robber(self, nums: List[int]) -> int:
        """
        Determine the maximum amount of money you can rob without robbing adjacent houses.

        Parameters:
            nums (List[int]): The amount of money in each house.

        Returns:
            int: The maximum amount of money that can be robbed.

        Raises:
            ValueError: If the nums list is empty.
        """
        if not nums:
            raise ValueError("Input list cannot be empty.")

        n = len(nums)
        if n == 1:
            return nums[0]

        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])

        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

        return dp[-1]

    def maximum_profit_stock(self, prices: List[int]) -> int:
        """
        Find the maximum profit you can achieve by making as many transactions as you like.

        Parameters:
            prices (List[int]): The price of the stock on each day.

        Returns:
            int: The maximum profit achievable.

        Raises:
            ValueError: If the prices list is empty.
        """
        if not prices:
            raise ValueError("Prices list cannot be empty.")

        profit = 0
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i - 1]
            if diff > 0:
                profit += diff
        return profit

    def decode_ways(self, s: str) -> int:
        """
        Determine the number of ways to decode a given digit string.

        'A' -> 1, 'B' -> 2, ..., 'Z' -> 26.

        Parameters:
            s (str): The encoded digit string.

        Returns:
            int: The number of ways to decode the string.

        Raises:
            ValueError: If the input string contains invalid characters.
        """
        if not s:
            return 0
        if any(c not in '0123456789' for c in s):
            raise ValueError("Input string must contain only digits.")

        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 1

        for i in range(1, n + 1):
            if s[i - 1] != '0':
                dp[i] += dp[i - 1]
            if i >= 2:
                two_digit = int(s[i - 2:i])
                if 10 <= two_digit <= 26:
                    dp[i] += dp[i - 2]

        return dp[n]

    def get_min_steps_to_one(self, n: int) -> int:
        """
        Find the minimum number of steps to reduce a number to one.

        You can perform the following operations:
            - If n is divisible by 3, divide by 3.
            - If n is divisible by 2, divide by 2.
            - Subtract 1.

        Parameters:
            n (int): The starting number.

        Returns:
            int: The minimum number of steps to reduce n to one.

        Raises:
            ValueError: If n is less than one.
        """
        if n < 1:
            raise ValueError("Input must be a positive integer.")

        dp = [0] * (n + 1)
        dp[1] = 0

        for i in range(2, n + 1):
            steps = [dp[i - 1] + 1]
            if i % 2 == 0:
                steps.append(dp[i // 2] + 1)
            if i % 3 == 0:
                steps.append(dp[i // 3] + 1)
            dp[i] = min(steps)

        return dp[n]

    def min_path_sum(self, grid: List[List[int]]) -> int:
        """
        Calculate the minimum path sum from top-left to bottom-right in a grid.

        You can only move either down or right at any point in time.

        Parameters:
            grid (List[List[int]]): The grid containing non-negative integers.

        Returns:
            int: The minimum path sum.

        Raises:
            ValueError: If the grid is empty or improperly formatted.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")

        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = grid[0][0]

        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] + grid[0][j]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]

        return dp[m - 1][n - 1]

    def count_palindromic_substrings(self, s: str) -> int:
        """
        Count the number of palindromic substrings in a given string.

        Parameters:
            s (str): The input string.

        Returns:
            int: The count of palindromic substrings.

        Raises:
            ValueError: If the input string is empty.
        """
        if not s:
            raise ValueError("Input string cannot be empty.")

        count = 0

        for i in range(len(s)):
            # Odd-length palindromes
            left, right = i, i
            while left >= 0 and right < len(s) and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1

            # Even-length palindromes
            left, right = i, i + 1
            while left >= 0 and right < len(s) and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1

        return count

    def maximum_product_subarray(self, nums: List[int]) -> int:
        """
        Find the contiguous subarray within an array which has the largest product.

        Parameters:
            nums (List[int]): The list of integers.

        Returns:
            int: The largest product of any contiguous subarray.

        Raises:
            ValueError: If nums is empty.
        """
        if not nums:
            raise ValueError("Input list cannot be empty.")

        max_prod = min_prod = result = nums[0]

        for num in nums[1:]:
            if num < 0:
                max_prod, min_prod = min_prod, max_prod

            max_prod = max(num, max_prod * num)
            min_prod = min(num, min_prod * num)

            result = max(result, max_prod)

        return result

    def integer_break(self, n: int) -> int:
        """
        Break a positive integer into the sum of at least two positive integers and maximize the product of those integers.

        Parameters:
            n (int): The integer to break.

        Returns:
            int: The maximum product possible.

        Raises:
            ValueError: If n is less than 2.
        """
        if n < 2:
            raise ValueError("Input must be at least 2.")

        dp = [0] * (n + 1)
        dp[1] = 1

        for i in range(2, n + 1):
            for j in range(1, i):
                dp[i] = max(dp[i], j * (i - j), j * dp[i - j])

        return dp[n]

    def regular_expression_matching(self, s: str, p: str) -> bool:
        """
        Implement regular expression matching with support for '.' and '*'.

        '.' Matches any single character.
        '*' Matches zero or more of the preceding element.

        Parameters:
            s (str): The input string.
            p (str): The pattern.

        Returns:
            bool: True if the string matches the pattern, False otherwise.

        Raises:
            ValueError: If either input is None.
        """
        if s is None or p is None:
            raise ValueError("Input string and pattern cannot be None.")

        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True

        for j in range(2, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 2]
                    if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                        dp[i][j] = dp[i][j] or dp[i - 1][j]
                else:
                    dp[i][j] = False

        return dp[m][n]

    def maximum_profit_with_cooldown(self, prices: List[int]) -> int:
        """
        Find the maximum profit with cooldown period after selling.

        Parameters:
            prices (List[int]): The list of stock prices.

        Returns:
            int: The maximum profit achievable.

        Raises:
            ValueError: If the prices list is empty.
        """
        if not prices:
            raise ValueError("Prices list cannot be empty.")

        n = len(prices)
        if n <= 1:
            return 0

        sell = [0] * n
        buy = [0] * n
        buy[0] = -prices[0]

        for i in range(1, n):
            buy[i] = max(buy[i - 1], sell[i - 2] - prices[i] if i >= 2 else -prices[i])
            sell[i] = max(sell[i - 1], buy[i - 1] + prices[i])

        return sell[-1]

    def longest_palindromic_subsequence(self, s: str) -> int:
        """
        Find the length of the longest palindromic subsequence in a string.

        Parameters:
            s (str): The input string.

        Returns:
            int: The length of the longest palindromic subsequence.

        Raises:
            ValueError: If the input string is empty.
        """
        if not s:
            raise ValueError("Input string cannot be empty.")

        n = len(s)
        dp = [[0] * n for _ in range(n)]

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

    def find_minimum_path_sum_triangle(self, triangle: List[List[int]]) -> int:
        """
        Find the minimum path sum from top to bottom in a triangle.

        Each step can move to adjacent numbers on the row below.

        Parameters:
            triangle (List[List[int]]): The triangle represented as a list of lists.

        Returns:
            int: The minimum path sum.

        Raises:
            ValueError: If the triangle is empty or improperly formatted.
        """
        if not triangle or not all(len(row) == idx + 1 for idx, row in enumerate(triangle)):
            raise ValueError("Invalid triangle structure.")

        dp = triangle[-1].copy()

        for i in range(len(triangle) - 2, -1, -1):
            for j in range(len(triangle[i])):
                dp[j] = triangle[i][j] + min(dp[j], dp[j + 1])

        return dp[0]

    def regular_expression_matching_with_plus(self, s: str, p: str) -> bool:
        """
        Extend regular expression matching to include '+' operator.

        '+' Matches one or more of the preceding element.

        Parameters:
            s (str): The input string.
            p (str): The pattern including '.' and '+'.

        Returns:
            bool: True if the string matches the pattern, False otherwise.

        Raises:
            ValueError: If either input is None.
        """
        if s is None or p is None:
            raise ValueError("Input string and pattern cannot be None.")

        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True

        # Initialize patterns with *
        for j in range(2, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == '*':
                    # Zero occurrence
                    dp[i][j] = dp[i][j - 2]
                    # One or more occurrence
                    if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                        dp[i][j] = dp[i][j] or dp[i - 1][j]
                elif p[j - 1] == '+':
                    # At least one occurrence
                    if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                        dp[i][j] = dp[i - 1][j] or dp[i - 2][j - 1]
                else:
                    dp[i][j] = False

        return dp[m][n]