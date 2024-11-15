#!/usr/bin/env python3
"""
String Algorithms and Data Structures Module

This module provides implementations of various string-related
data structures and algorithms, ranging from basic to advanced levels.

Author: Generalmodelai-agent
Date: 2024-10-15
"""

from typing import List, Dict, Optional, Tuple


class KMPMatcher:
    """
    Knuth-Morris-Pratt (KMP) algorithm for substring search.
    """

    def __init__(self, pattern: str) -> None:
        """
        Initializes the KMP matcher with the given pattern.

        Args:
            pattern (str): The pattern to search for.
        """
        self.pattern = pattern
        self.lps = self._compute_lps(pattern)

    @staticmethod
    def _compute_lps(pattern: str) -> List[int]:
        """
        Computes the Longest Prefix Suffix (LPS) array for the pattern.

        Args:
            pattern (str): The pattern string.

        Returns:
            List[int]: The LPS array.
        """
        lps = [0] * len(pattern)
        length = 0  # length of the previous longest prefix suffix

        for i in range(1, len(pattern)):
            while length > 0 and pattern[i] != pattern[length]:
                length = lps[length - 1]
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
            else:
                lps[i] = length
        return lps

    def search(self, text: str) -> List[int]:
        """
        Searches for occurrences of the pattern in the given text.

        Args:
            text (str): The text to search within.

        Returns:
            List[int]: A list of starting indices where pattern is found.
        """
        indices = []
        m = len(self.pattern)
        n = len(text)
        i = j = 0  # indexes for text and pattern

        while i < n:
            if text[i] == self.pattern[j]:
                i += 1
                j += 1
                if j == m:
                    indices.append(i - j)
                    j = self.lps[j - 1]
            else:
                if j != 0:
                    j = self.lps[j - 1]
                else:
                    i += 1
        return indices


class RabinKarpMatcher:
    """
    Rabin-Karp algorithm for substring search.
    """

    def __init__(self, pattern: str, prime: int = 101) -> None:
        """
        Initializes the Rabin-Karp matcher with the given pattern.

        Args:
            pattern (str): The pattern to search for.
            prime (int, optional): A prime number for hashing. Defaults to 101.
        """
        self.pattern = pattern
        self.pattern_length = len(pattern)
        self.prime = prime
        self.pattern_hash = self._hash(pattern)
        self.high_order = pow(self.prime, self.pattern_length - 1)

    def _hash(self, s: str) -> int:
        """
        Computes the hash value of a string.

        Args:
            s (str): The string to hash.

        Returns:
            int: The hash value.
        """
        h = 0
        for char in s:
            h = h * self.prime + ord(char)
        return h

    def search(self, text: str) -> List[int]:
        """
        Searches for occurrences of the pattern in the given text.

        Args:
            text (str): The text to search within.

        Returns:
            List[int]: A list of starting indices where pattern is found.
        """
        indices = []
        n = len(text)
        if self.pattern_length > n:
            return indices

        current_hash = self._hash(text[:self.pattern_length])

        for i in range(n - self.pattern_length + 1):
            if current_hash == self.pattern_hash:
                if text[i:i + self.pattern_length] == self.pattern:
                    indices.append(i)
            if i < n - self.pattern_length:
                current_hash = (current_hash - ord(text[i]) * self.high_order) * self.prime + ord(text[i + self.pattern_length])
        return indices


class TrieNode:
    """
    Node in a Trie (prefix tree).
    """

    def __init__(self) -> None:
        """
        Initializes a Trie node.
        """
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = False


class Trie:
    """
    Trie data structure for efficient string storage and retrieval.
    """

    def __init__(self) -> None:
        """
        Initializes the Trie.
        """
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the Trie.

        Args:
            word (str): The word to insert.
        """
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end_of_word = True

    def search(self, word: str) -> bool:
        """
        Searches for a word in the Trie.

        Args:
            word (str): The word to search for.

        Returns:
            bool: True if the word exists, False otherwise.
        """
        current = self.root
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        return current.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """
        Checks if any word in the Trie starts with the given prefix.

        Args:
            prefix (str): The prefix to check.

        Returns:
            bool: True if any word starts with the prefix, False otherwise.
        """
        current = self.root
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]
        return True


class SuffixTreeNode:
    """
    Node in a Suffix Tree.
    """

    def __init__(self):
        """
        Initializes a Suffix Tree node.
        """
        self.children: Dict[str, 'SuffixTreeNode'] = {}
        self.indexes: List[int] = []


class SuffixTree:
    """
    Suffix Tree data structure for efficient substring queries.
    """

    def __init__(self, text: str) -> None:
        """
        Builds the Suffix Tree for the given text.

        Args:
            text (str): The text to build the suffix tree for.
        """
        self.root = SuffixTreeNode()
        self.text = text
        self._build_suffix_tree()

    def _build_suffix_tree(self) -> None:
        """
        Builds the suffix tree by inserting all suffixes.
        """
        for i in range(len(self.text)):
            current = self.root
            for char in self.text[i:]:
                if char not in current.children:
                    current.children[char] = SuffixTreeNode()
                current = current.children[char]
                current.indexes.append(i)

    def search(self, pattern: str) -> List[int]:
        """
        Searches for a pattern in the suffix tree.

        Args:
            pattern (str): The pattern to search for.

        Returns:
            List[int]: List of starting indices where pattern is found.
        """
        current = self.root
        for char in pattern:
            if char not in current.children:
                return []
            current = current.children[char]
        return current.indexes


def longest_palindromic_substring(s: str) -> str:
    """
    Finds the longest palindromic substring in the given string using Manacher's Algorithm.

    Args:
        s (str): The input string.

    Returns:
        str: The longest palindromic substring.
    """
    if not s:
        return ""

    # Transform the string to handle even-length palindromes
    transformed = '#' + '#'.join(s) + '#'
    n = len(transformed)
    lps = [0] * n
    center = right = 0
    max_len = 0
    center_index = 0

    for i in range(n):
        mirror = 2 * center - i
        if i < right:
            lps[i] = min(right - i, lps[mirror])
        # Expand around center i
        a = i + lps[i] + 1
        b = i - lps[i] - 1
        while a < n and b >= 0 and transformed[a] == transformed[b]:
            lps[i] += 1
            a += 1
            b -= 1
        # Update center and right boundary
        if i + lps[i] > right:
            center = i
            right = i + lps[i]
        # Track maximum palindrome length
        if lps[i] > max_len:
            max_len = lps[i]
            center_index = i

    start = (center_index - max_len) // 2
    return s[start:start + max_len]


def longest_common_substring(s1: str, s2: str) -> str:
    """
    Finds the longest common substring between two strings using Dynamic Programming.

    Args:
        s1 (str): First string.
        s2 (str): Second string.

    Returns:
        str: The longest common substring.
    """
    if not s1 or not s2:
        return ""

    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    max_len = 0
    end_idx = 0

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_idx = i
            else:
                dp[i][j] = 0

    return s1[end_idx - max_len:end_idx]


def reverse_string(s: str) -> str:
    """
    Reverses the given string.

    Args:
        s (str): The string to reverse.

    Returns:
        str: The reversed string.
    """
    return s[::-1]


def is_anagram(s1: str, s2: str) -> bool:
    """
    Checks if two strings are anagrams.

    Args:
        s1 (str): First string.
        s2 (str): Second string.

    Returns:
        bool: True if anagrams, False otherwise.
    """
    if len(s1) != len(s2):
        return False
    count: Dict[str, int] = {}
    for char in s1:
        count[char] = count.get(char, 0) + 1
    for char in s2:
        if char not in count:
            return False
        count[char] -= 1
        if count[char] < 0:
            return False
    return True


def find_all_unique_substrings(s: str) -> List[str]:
    """
    Finds all unique substrings of the given string.

    Args:
        s (str): The input string.

    Returns:
        List[str]: A list of unique substrings.
    """
    substrings = set()
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n + 1):
            substrings.add(s[i:j])
    return list(substrings)


def find_all_unique_permutations(s: str) -> List[str]:
    """
    Finds all unique permutations of the given string.

    Args:
        s (str): The input string.

    Returns:
        List[str]: A list of unique permutations.
    """
    results: List[str] = []
    s_sorted = sorted(s)
    used = [False] * len(s_sorted)
    permutation: List[str] = []

    def backtrack() -> None:
        if len(permutation) == len(s_sorted):
            results.append(''.join(permutation))
            return
        for i in range(len(s_sorted)):
            if used[i]:
                continue
            if i > 0 and s_sorted[i] == s_sorted[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            permutation.append(s_sorted[i])
            backtrack()
            used[i] = False
            permutation.pop()

    backtrack()
    return results


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes the Levenshtein distance between two strings.

    Args:
        s1 (str): First string.
        s2 (str): Second string.

    Returns:
        int: The Levenshtein distance.
    """
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        current_row = [i]
        for j, c2 in enumerate(s2, 1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compress_string(s: str) -> str:
    """
    Compresses the string using basic run-length encoding.

    Args:
        s (str): The input string.

    Returns:
        str: The compressed string.
    """
    if not s:
        return ""

    compressed = []
    count = 1
    prev = s[0]

    for char in s[1:]:
        if char == prev:
            count += 1
        else:
            compressed.append(f"{prev}{count}")
            prev = char
            count = 1
    compressed.append(f"{prev}{count}")
    compressed_str = ''.join(compressed)
    return compressed_str if len(compressed_str) < len(s) else s


def decode_compressed_string(s: str) -> str:
    """
    Decodes a run-length encoded string.

    Args:
        s (str): The compressed string.

    Returns:
        str: The original string.
    """
    decoded = []
    count = 0

    for char in s:
        if char.isdigit():
            count = count * 10 + int(char)
        else:
            if count == 0:
                count = 1
            decoded.append(char * count)
            count = 0
    return ''.join(decoded)


if __name__ == "__main__":
    # Example Usage

    # KMP Matcher
    kmp = KMPMatcher("pattern")
    print("KMP Search Indices:", kmp.search("this is a pattern in a patterned string"))

    # Rabin-Karp Matcher
    rk = RabinKarpMatcher("pattern")
    print("Rabin-Karp Search Indices:", rk.search("this is a pattern in a patterned string"))

    # Trie Operations
    trie = Trie()
    trie.insert("hello")
    trie.insert("hell")
    trie.insert("heaven")
    print("Trie Search 'hell':", trie.search("hell"))
    print("Trie Search 'heaven':", trie.search("heaven"))
    print("Trie Search 'heavy':", trie.search("heavy"))
    print("Trie Starts With 'hea':", trie.starts_with("hea"))

    # Suffix Tree Search
    suffix_tree = SuffixTree("banana")
    print("Suffix Tree Search 'ana':", suffix_tree.search("ana"))

    # Longest Palindromic Substring
    print("Longest Palindromic Substring:", longest_palindromic_substring("babad"))

    # Longest Common Substring
    print("Longest Common Substring:", longest_common_substring("abcdef", "zabxcdefy"))

    # Reverse String
    print("Reversed String:", reverse_string("hello"))

    # Anagram Check
    print("Are 'listen' and 'silent' anagrams?:", is_anagram("listen", "silent"))

    # All Unique Substrings
    print("All Unique Substrings:", find_all_unique_substrings("abc"))

    # All Unique Permutations
    print("All Unique Permutations:", find_all_unique_permutations("aab"))

    # Levenshtein Distance
    print("Levenshtein Distance:", levenshtein_distance("kitten", "sitting"))

    # String Compression
    print("Compressed String:", compress_string("aaabccdddde"))
    print("Decoded String:", decode_compressed_string("a3b1c2d4e1"))